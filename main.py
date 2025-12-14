import argparse
import os
import datetime
import time
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import datasets
from resnet import ResNet18
import query_strategies
from utils import get_splits, lab_conv
from llm import load_prompts
import clip


parser = argparse.ArgumentParser("LaSeR")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tinyimagenet'])
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")
parser.add_argument('--batch-size', type=int, default=128)
# optimization
parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=300)
parser.add_argument('--max-query', type=int, default=10) 
parser.add_argument('--query-batch', type=int, default=1500)
parser.add_argument('--stepsize', type=int, default=60)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
parser.add_argument('--query-strategy', type=str, default='laser', choices=['random', 'laser'])
parser.add_argument('--K-td', type=int, default=4)  # number of generated text descriptions (15/4/2 for the CIFAR10/CIFAR100/Tiny-ImageNet datasets, respectively)
# model
parser.add_argument('--model', type=str, default='resnet18')
# misc
parser.add_argument('--eval-freq', type=int, default=300)
parser.add_argument('--gpu', type=str, default='0') 
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
# openset
parser.add_argument('--is-filter', type=bool, default=True)
parser.add_argument('--known-class', type=int, default=20)  # mismatch ratio (@20% - 2/20/40 for the CIFAR10/CIFAR100/Tiny-ImageNet datasets, respectively)

args = parser.parse_args()

def main():
    seed = 1
    delta = 0 # llm factor
    all_accuracies = []
    all_precisions = []
    all_recalls = []

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    if use_gpu:
        print("Using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Using CPU")

    knownclass = get_splits(args.dataset, seed, args.known_class)  # ground truth labels
    N_conf = 2*args.known_class
    print("-------------------- Cold Start Problem --------------------")
    print("Fraction of known classes amoung classes:", knownclass)
    print("Number of text descriptions generated per class:", args.K_td)
    print("Number of LLM-generated irrelevant classes:", N_conf)
    print("------------------------------------------------------------")

    print("Creating dataset: {}".format(args.dataset))
    
    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, knownclass=knownclass,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter
    )

    unlabeledloader = dataset.unlabeledloader  # unlabeled indices
    irrelevantList = []
    labeled_ind_train = []
    relevantclass, irrelevantclass, relevant_generated, irrelevant_generated = [], [], [], []
    unlabeled_ind_train = dataset.unlabeled_ind_train
    class_to_idx = dataset.class_to_idx
    idx_to_class = dataset.idx_to_class
    
    print("Unlabeled query batches:", len(unlabeledloader))

    # Load LLM prompts from JSON
    llm_prompts = load_prompts()
    # initialize VLM
    device = "cuda" if use_gpu else "cpu"
    vlm, _ = clip.load('ViT-B/32', device)

    print("Creating model: {}".format(args.model))
    start_time = time.time()

    Acc = {} 
    Err = {}
    Precision = {}
    Recall = {}

    for query in tqdm(range(args.max_query)):  
        # Query Sampling
        queryIndex = []
        if args.query_strategy == "laser":
            queryIndex, irrelevantIndex, relevantclass, irrelevantclass, relevant_generated, irrelevant_generated, Precision[query], Recall[query] = query_strategies.laser_sampling(args, unlabeledloader, len(labeled_ind_train), knownclass, relevantclass, irrelevantclass, relevant_generated, irrelevant_generated, N_conf, delta, llm_prompts, vlm, use_gpu, idx_to_class, class_to_idx, query)

        # Update labeled, unlabeled and invalid set
        unlabeled_ind_train = list(set(unlabeled_ind_train)-set(queryIndex))
        labeled_ind_train = list(labeled_ind_train) + list(queryIndex)
        irrelevantList = list(irrelevantList) + list(irrelevantIndex)

        print("Query Strategy: "+args.query_strategy+" | Query Budget: "+str(args.query_batch)+" | Valid Query Nums: "+str(len(queryIndex))+" | Query Precision: "+str(Precision[query])+" | Query Recall: "+str(Recall[query])+" | Training Nums: "+str(len(labeled_ind_train)))

        # Recreate dataloaders with updated labeled/unlabeled splits
        dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train
        )
        trainloader, testloader, unlabeledloader = dataset.trainloader, dataset.testloader, dataset.unlabeledloader

        # Model initialization
        model = ResNet18(num_classes=dataset.num_classes)
        print(dataset.num_classes)
        if use_gpu:
            model = nn.DataParallel(model).cuda()

        # Define losses and optimizers
        criterion_xent = nn.CrossEntropyLoss()   # standard classification loss
        optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        if args.stepsize > 0:
            scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

        # Model training 
        for epoch in tqdm(range(args.max_epoch)):
            train_model(model, criterion_xent, optimizer_model, trainloader, use_gpu, knownclass)

            if args.stepsize > 0:
                scheduler.step()

            # Evaluate periodically
            if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
                print("==> Test")
                accuracy, error = test(model, testloader, use_gpu, knownclass)
                print("Classifier | Accuracy (%): {}\t Error rate (%): {}".format(accuracy, error))

        # Record results
        acc, err = test(model, testloader, use_gpu, knownclass)
        Acc[query], Err[query] = float(acc), float(err)

    all_accuracies.append(Acc)
    all_precisions.append(Precision)
    all_recalls.append(Recall)
    print("Accuracies", all_accuracies)
    print("Precisions", all_precisions)
    print("Recalls", all_recalls)
   
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train_model(model, criterion_xent, optimizer_model, trainloader, use_gpu, relevantclass):
    model.train()

    for batch_idx, (index, (data, labels))  in enumerate(trainloader):
        labels = lab_conv(relevantclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs = model(data)  # Forward pass
        loss_xent = criterion_xent(outputs, labels)  # Compute loss
        loss = loss_xent 
        # Backpropagation
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()


def test(model, testloader, use_gpu, relevantclass):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for index, (data, labels) in testloader:
            labels = lab_conv(relevantclass, labels)
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            predictions = outputs.argmax(dim=1) # pick class with max logit
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct * 100. / total
    error = 100. - accuracy
    return accuracy, error

if __name__ == '__main__':
    main()