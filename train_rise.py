import os
import argparse
import torch
import clip
from torch import nn
from torch.nn import functional as F
from data import data_helper
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
from datetime import datetime
from timm.models import create_model

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch CLIP distillation")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--Domain_ID", default=['sketch', 'photo', 'cartoon', 'art_painting'])
    parser.add_argument("--classes", default=["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscale")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", default="resnetv2_50x1_bit.goog_in21k_ft_in1k", help="Which network to use")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='', help="Used by the logger to save logs")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--output_folder", default='run1', help="folder where to save results file")
    parser.add_argument("--output_file_name", default='.txt', help="results file name")
    parser.add_argument("--data_path", default='', help="path of the dataset")

    return parser.parse_args()

class Trainer:
    def __init__(self, args, device, tt, ww1, ww2, ww3, target_name):
        self.args = args
        self.device = device

        self.clip_model, _ = clip.load(self.args.CLIP, device=self.device)
        if self.args.dataset == "Terra":
            print("please load your finetuned CLIP weight here")
            # model_weights = torch.load("/path/finetuned_clip")
            # self.clip_model.load_state_dict(model_weights)
        self.text_feature_dim = 512
        # ---CLIP prompt engineering
        t1 = torch.cat([clip.tokenize(f"itap of a {c}.") for c in self.args.classes]).to(self.device)
        t2 = torch.cat([clip.tokenize(f"a bad photo of the {c}.") for c in self.args.classes]).to(self.device)
        t3 = torch.cat([clip.tokenize(f"a origami {c}.") for c in self.args.classes]).to(self.device)
        t4 = torch.cat([clip.tokenize(f"a photo of the large {c}.") for c in self.args.classes]).to(self.device)
        t5 = torch.cat([clip.tokenize(f"a {c} in a video game.") for c in self.args.classes]).to(self.device)
        t6 = torch.cat([clip.tokenize(f"art of the {c}.") for c in self.args.classes]).to(self.device)
        t7 = torch.cat([clip.tokenize(f"a photo of the small {c}.") for c in self.args.classes]).to(self.device)

        text_list = []
        if args.dataset == "Terra":
            self.text_anchor = ['bright photo', 'corrupted photo', 'dark photo', 'good photo']
        elif args.dataset == "VLCS":
            self.text_anchor = ['bright photo', 'corrupted photo', 'dark photo', 'good photo']
        else:
            self.text_anchor = args.source
        for source in self.text_anchor:
            text_list.append(torch.cat([clip.tokenize(f"a {source} of a {c}") for c in self.args.classes]).to(device))

        text_token_list = []
        with torch.no_grad():
            self.clip_model.eval()
            text1 = self.clip_model.encode_text(t1)
            text2 = self.clip_model.encode_text(t2)
            text3 = self.clip_model.encode_text(t3)
            text4 = self.clip_model.encode_text(t4)
            text5 = self.clip_model.encode_text(t5)
            text6 = self.clip_model.encode_text(t6)
            text7 = self.clip_model.encode_text(t7)

            for text in text_list:
                text_token_list.append(self.clip_model.encode_text(text))

            self.text_features_ems = (text1 + text2 + text3 + text4 + text5 + text6 + text7) / 7.0
            self.CLIP_text_features_ems_before_norm = self.text_features_ems.clone().detach().type(torch.float32).to(self.device)
            self.text_features_ems /= self.text_features_ems.norm(dim=-1, keepdim=True)
        self.text_compare_teacher = torch.zeros(self.args.n_classes, len(self.text_anchor), self.text_feature_dim).to(self.device)

        for i in range(self.args.n_classes):
            for j in range(len(self.text_anchor)):
                self.text_compare_teacher[i, j, :] = text_token_list[j][i]

        model = create_model(self.args.network, pretrained=True, num_classes=self.args.n_classes)
        model.fc.weight.data = self.text_features_ems.data.float().clone().detach()
        self.model = model.to(self.device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        self.target_loader = data_helper.get_val_dataloader(args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (
        len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(self.model, args.epochs, args.learning_rate, args.train_all,
                                                                 nesterov=False)
        self.current_epoch = 0
        self.distill_weight = ww1
        self.classification_weight = ww2
        self.distance_weight = ww3
        self.T = tt
        self.target_name = target_name
        print("Loss weight: distill %.2f, cls %.2f, RD %.2f. T: %.2f" % (
            self.distill_weight, self.classification_weight, self.distance_weight, self.T))

    def _do_epoch(self):
        softmax = nn.Softmax(dim=1).cuda()
        criterion = nn.CrossEntropyLoss()
        cosine_sim_loss = torch.nn.CosineEmbeddingLoss()
        self.model.train()

        for it, ((data, class_l), d_idx) in enumerate(self.source_loader):
            data, class_l, d_idx = data.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            bs = data.shape[0]

            # Calculate features
            with torch.no_grad():
                self.clip_model.eval()
                CLIP_image_features = self.clip_model.encode_image(data)
            CLIP_image_features /= CLIP_image_features.norm(dim=-1, keepdim=True)
            teacher_logits = (100.0 * CLIP_image_features @ self.text_features_ems.T).type(torch.float32)

            self.optimizer.zero_grad()
            student_embedding, class_logit = self.model(data)

            # --- classification loss
            supervised_loss = criterion(class_logit, class_l)
            # --- distillation loss
            kl_loss = F.kl_div(F.log_softmax(class_logit / self.T, dim=1),
                               F.softmax(teacher_logits / self.T, dim=1),
                               reduction='batchmean') * self.T * self.T

            # --- absolute distance loss
            clip_text_embedding_instance = torch.zeros(student_embedding.shape[0], self.text_feature_dim).to(self.device)
            for i in range(bs):
                clip_text_embedding_instance[i, :] = self.text_features_ems[class_l[i], :]
            cosine_sim_label = torch.ones(student_embedding.shape[0]).to(self.device)
            text_embed_sim_loss = cosine_sim_loss(F.normalize(student_embedding, dim=-1), clip_text_embedding_instance, cosine_sim_label)

            # --- relative distance loss
            dist_teacher = torch.zeros(bs, len(self.text_anchor)).to(self.device)
            dist_student = torch.zeros(bs, len(self.text_anchor)).to(self.device)
            for pair1 in range(bs):
                tmp_anchor_feat_student = student_embedding[pair1, :]
                gt = class_l[pair1]
                tmp_anchor_feat_teacher = self.CLIP_text_features_ems_before_norm[gt]
                compare_feat = self.text_compare_teacher[gt]
                dist_teacher[pair1, :] = F.cosine_similarity(tmp_anchor_feat_teacher.repeat(len(self.text_anchor), 1), compare_feat)
                dist_student[pair1, :] = F.cosine_similarity(tmp_anchor_feat_student.repeat(len(self.text_anchor), 1), compare_feat)
            dist_teacher = softmax(dist_teacher)
            dist_student = softmax(dist_student)
            domain_feature_relation_loss = F.mse_loss(dist_student, dist_teacher) * 10.0

            class_probs = class_logit.softmax(dim=-1)
            _, cls_pred = class_probs.max(dim=1)
            loss = kl_loss * self.distill_weight \
                   + supervised_loss * self.classification_weight \
                   + text_embed_sim_loss * self.distance_weight + domain_feature_relation_loss * self.distance_weight
            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            {
                             "CE": supervised_loss.item(), "Hint": kl_loss.item(),
                             "AD": text_embed_sim_loss.item(), "RD": (domain_feature_relation_loss*1000).item()
                             },
                            {"class": torch.sum(cls_pred == class_l.data).item()
                            },
                            data.shape[0])
            del loss, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        class_correct = 0
        for it, ((data, class_l), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            student_embedding, student_logits = self.model(data)
            similarity = student_logits.softmax(dim=-1)
            if self.args.dataset == "Terra":
                student_embedding /= student_embedding.norm(dim=-1, keepdim=True)
                student_logits_clip = (100.0 * student_embedding @ self.text_features_ems.T.type(torch.float32)).type(torch.float32)
                similarity_clip = student_logits_clip.softmax(dim=-1)
                similarity += similarity_clip
            _, cls_pred = similarity.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.logger.new_epoch(self.scheduler.get_last_lr())
            self._do_epoch()
            self.scheduler.step()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        self.logger.save_best(test_res[idx_best], test_res.max())
        with open(self.args.output_file_name, 'a') as ff:
            ff.write(
                f'target domain {self.target_name}, t : {self.T}, w1: {self.distill_weight}, w2: {self.classification_weight}, w3: {self.distance_weight}')
            ff.write('\n')
            ff.write(f'Best val {val_res.max()}, corresponding test {test_res[idx_best]} - best test: {test_res.max()}, best epoch: {idx_best}')
            ff.write('\n')
            ff.write('\n')
        return self.logger, self.model


def train_with_sweep():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:"+args.GPU_num if torch.cuda.is_available() else "cpu")
    select_txt = os.path.join(os.getcwd(), 'data', 'hp_search', args.dataset + '.txt')
    print("parameter search space: ")
    with open(select_txt, 'r') as ff:
        lines = ff.readlines()
        print(lines)


    if args.dataset == "PACS":
        args.Domain_ID = ['art_painting', 'sketch', 'photo', 'cartoon']
        args.classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        args.n_classes = 7
        args.n_domain = 4
    elif args.dataset == "VLCS":
        args.Domain_ID = ["LABELME", "SUN", "VOC", "CALTECH"]
        args.classes = ["bird", "car", "chair", "dog", "person"]
        args.n_classes = 5
        args.n_domain = 4
    elif args.dataset == "Terra":
        args.Domain_ID = ["location_100", "location_38", "location_43", "location_46"]
        args.classes = ["bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit", "raccoon", "squirrel"]
        args.n_classes = 10
        args.n_domain = 4
        args.learning_rate = 0.002
    elif args.dataset == "Officehome":
        args.Domain_ID = ['Clipart', 'Art', 'RealWorld', 'Product']
        args.classes = ["Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike", "Bottle", "Bucket", "Calculator",
                        "Calendar", "Candles", "Chair", "Clipboards", "Computer", "Couch", "Curtains", "Desk_Lamp",
                        "Drill", "Eraser", "Exit_Sign", "Fan", "File_Cabinet", "Flipflops", "Flowers", "Folder", "Fork",
                        "Glasses", "Hammer", "Helmet", "Kettle", "Keyboard", "Knives", "Lamp_Shade", "Laptop", "Marker",
                        "Monitor", "Mop", "Mouse", "Mug", "Notebook", "Oven", "Pan", "Paper_Clip", "Pen", "Pencil",
                        "Postit_Notes", "Printer", "Push_Pin", "Radio", "Refrigerator", "Ruler", "Scissors",
                        "Screwdriver", "Shelf", "Sink", "Sneakers", "Soda", "Speaker", "Spoon", "Table", "Telephone",
                        "Toothbrush", "Toys", "Trash_Can", "TV", "Webcam"]
        args.n_classes = 65
        args.n_domain = 4
    else:
        raise NotImplementedError

    for domain in args.Domain_ID:
        args.target = domain
        args.source = args.Domain_ID.copy()
        args.source.remove(args.target)
        print("Training {} on source domains:".format(args.dataset))
        print(*args.source, sep=",")
        print("Test on target domains:")
        print(args.target)

        now = datetime.now().strftime("%m-%d-%y_%H:%M:%S")
        output_file_name = now + '_' + args.dataset + '_' + args.target + '.txt'
        output_folder = os.path.join(os.getcwd(), 'results', args.output_folder)
        if os.path.exists(output_folder):
            pass
        else:
            os.makedirs(output_folder)
        args.output_file_name = os.path.join(output_folder, output_file_name)
        print("output results are saved at: {}".format(args.output_file_name))

        for line in lines:
            eles = line.strip().split(' ')
            tt = float(eles[0])
            w1 = float(eles[1])
            w2 = float(eles[2])
            w3 = float(eles[3])
            trainer = Trainer(args, device, tt, w1, w2, w3, args.target)
            trainer.do_training()

if __name__ == "__main__":
    train_with_sweep()
