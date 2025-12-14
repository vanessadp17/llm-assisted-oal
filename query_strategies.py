
import numpy as np
import torch
import numpy as np
from utils import lab_conv, get_labels, get_descriptions, image_embedding, text_embedding


def laser_sampling(args, unlabeledloader, Len_labeled_ind_train, knownclass, relevantclass, irrelevantclass, relevant_prior, irrelevant_prior, N_conf, delta, prompts, vlm, use_gpu, idx_to_class, class_to_idx, query):
    labelArr, queryIndex, relevance_list = [], [], []
    precision, recall = 0, 0
    device = "cuda" if use_gpu else "cpu"
    task = prompts['task_description'][args.dataset][str(args.known_class)]

    # LLM generated labels
    if query == 0:
        relevant_labels = get_labels(prompts['relevant_labels'], N_conf, task=task)
        irrelevant_labels = get_labels(prompts['irrelevant_labels'], N_conf, task=task, rLabelsArr=relevant_labels)
    else:
        N = N_conf-len(relevantclass)*2
        relevant_class = [idx_to_class[i] for i in relevantclass]
        relevant_labels = get_labels(prompts['relevant_labels_with_prior'], N, task, True, relevant_class, relevant_prior)
        irrelevant_labels = get_labels(prompts['irrelevant_labels'], N, task=task, rLabelsArr=relevant_class)
    # LLM generated descriptions
    relevant_descriptions = get_descriptions(prompts['class_descriptions'], relevant_labels, args.K_td)
    irrelevant_descriptions = get_descriptions(prompts['class_descriptions'], irrelevant_labels, args.K_td)

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        # Compute VLM-based relevance score
        image_features = image_embedding(data, vlm, device)
        relevant_text_features = text_embedding(relevant_descriptions, vlm, device)
        irrelevant_text_features = text_embedding(irrelevant_descriptions, vlm, device)

        # Compute cosine similarities
        max_cos_r = (image_features @ relevant_text_features.T).max(dim=1).values  # returns Tensor
        avg_cos_irr = (image_features @ irrelevant_text_features.T).mean(dim=1)

        # Compute relevance score
        relevance_score = max_cos_r - avg_cos_irr

        # Collect data for later analysis
        relevance_list.append(relevance_score.cpu().data)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    print("relevance list length: ", len(relevance_list))
    relevance_list = torch.cat(relevance_list).cpu()  # Make into 1 Tensor
    relevance_list = (1 - delta)*relevance_list

    labelArr = torch.tensor(labelArr)
    queryIndex = torch.tensor(queryIndex)

    # Annotations step
    rem = args.query_batch
    temp_relevance_list = relevance_list.clone()
    selected_idx = []
    selected_gt = []

    while rem > 0:
        sorted_idx = torch.argmax(temp_relevance_list).item()
        q_idx = int(queryIndex[sorted_idx])
        s_gt = int(labelArr[sorted_idx])
        # Determine query cost
        class_cost, image_cost = 0, 1
        if s_gt in knownclass and s_gt not in relevantclass: # relevant ground truth
            class_cost = 1
        elif s_gt not in knownclass and s_gt not in irrelevantclass: # irrelevant ground truth
            class_cost = 1
        total_cost = class_cost + image_cost
        if total_cost > rem:
            break

        if s_gt in knownclass and s_gt not in relevantclass: # relevant ground truth
            relevantclass.append(s_gt)
        elif s_gt not in knownclass and s_gt not in irrelevantclass: # irrelevant ground truth
            irrelevantclass.append(s_gt)

        selected_idx.append(q_idx)
        selected_gt.append(s_gt)
        rem -= total_cost # image labeling cost
        temp_relevance_list[sorted_idx] = float('-inf')

    selected_gt = np.array(selected_gt)
    selected_idx = np.array(selected_idx)

    # Compute evaluation metrics 
    precision = len(np.where(selected_gt < args.known_class)[0]) / len(selected_gt)
    recall = (len(np.where(selected_gt < args.known_class)[0]) + Len_labeled_ind_train) / (
                len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return selected_idx[np.where(selected_gt < args.known_class)[0]], selected_idx[np.where(selected_gt >= args.known_class)[0]], relevantclass, irrelevantclass, relevant_labels, irrelevant_labels, precision, recall