import torch
import clip
import torch.nn.functional as F
from llm import prepare_label_prompt, prepare_description_prompt, generate_response

def get_splits(dataset, seed, mismatch):
    if dataset == 'cifar10':
        if seed == 1:
            shuffled_list = [5, 3, 2, 6]
    elif dataset == 'cifar100':
        if seed == 1:
            shuffled_list = [0, 83, 10, 51, 61, 57, 53, 26, 45, 91, 13, 8, 90, 81, 5, 84, 20, 94, 40, 87, 6, 7, 14, 18, 24, 99, 79, 80, 75, 66, 1, 36, 65, 93, 78, 70, 92, 82, 62, 54]
    knownclass = shuffled_list[:mismatch]
    return knownclass

def lab_conv(knownclass, label):
    knownclass = sorted(knownclass)
    label_convert = torch.zeros(len(label), dtype=int)
    for j in range(len(label)):
        for i in range(len(knownclass)):

            if label[j] == knownclass[i]:
                label_convert[j] = int(knownclass.index(knownclass[i]))
                break
            else:
                label_convert[j] = int(len(knownclass))     
    return label_convert

def get_labels(prompt, N, task=None, with_prior=False, rLabelsArr=None, llmLabelsArr=None):
    if with_prior:
        prompt = prepare_label_prompt(prompt, N, task=task, classArr=rLabelsArr, llmArr=llmLabelsArr)
    else:
        prompt = prepare_label_prompt(prompt, N, task=task, classArr=rLabelsArr)
    response = generate_response(prompt, "labels", N)
    return response['labels']

def get_descriptions(descPrompt, labels, K):
    descPrompt = prepare_description_prompt(descPrompt, K, labels)
    descResponse = generate_response(descPrompt, "descriptions", K)
    return descResponse['descriptions']

def image_embedding(data, model, device):
    data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
    data = data.to(device)
    with torch.no_grad():
        image_features = model.encode_image(data)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

def text_embedding(class_text, model, device):
    class_list = [desc for descs in class_text.values() for desc in descs]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_list]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=1, keepdim=True)
    return text_features

