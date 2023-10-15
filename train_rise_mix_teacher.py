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
    parser.add_argument("--learning_rate_2", "-l2", type=float, default=.001, help="Learning rate")
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

        self.clip_model_ViT, _ = clip.load("ViT-B/16", device=self.device)
        self.clip_model_RN, self.clip_transform = clip.load("RN101", device=self.device)
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
            with torch.no_grad():
                self.clip_model_ViT.eval()
                text1_ViT = self.clip_model_ViT.encode_text(t1)
                text2_ViT = self.clip_model_ViT.encode_text(t2)
                text3_ViT = self.clip_model_ViT.encode_text(t3)
                text4_ViT = self.clip_model_ViT.encode_text(t4)
                text5_ViT = self.clip_model_ViT.encode_text(t5)
                text6_ViT = self.clip_model_ViT.encode_text(t6)
                text7_ViT = self.clip_model_ViT.encode_text(t7)
                self.text_features_ems_ViT = (text1_ViT + text2_ViT + text3_ViT + text4_ViT + text5_ViT + text6_ViT + text7_ViT) / 7.0
                self.CLIP_text_features_ems_before_norm = self.text_features_ems_ViT.clone().detach().type(torch.float32).to(self.device)
                self.text_features_ems_ViT /= self.text_features_ems_ViT.norm(dim=-1, keepdim=True)

                self.clip_model_ViT.eval()
                text1_RN = self.clip_model_RN.encode_text(t1)
                text2_RN = self.clip_model_RN.encode_text(t2)
                text3_RN = self.clip_model_RN.encode_text(t3)
                text4_RN = self.clip_model_RN.encode_text(t4)
                text5_RN = self.clip_model_RN.encode_text(t5)
                text6_RN = self.clip_model_RN.encode_text(t6)
                text7_RN = self.clip_model_RN.encode_text(t7)
                self.text_features_ems_RN = (text1_RN + text2_RN + text3_RN + text4_RN + text5_RN + text6_RN + text7_RN) / 7.0
                self.text_features_ems_RN /= self.text_features_ems_RN.norm(dim=-1, keepdim=True)

            for text in text_list:
                text_token_list.append(self.clip_model_ViT.encode_text(text))

        self.text_compare_teacher = torch.zeros(self.args.n_classes, len(self.text_anchor), self.text_feature_dim).to(self.device)
        for i in range(self.args.n_classes):
            for j in range(len(self.text_anchor)):
                self.text_compare_teacher[i, j, :] = text_token_list[j][i]

        model1 = create_model(self.args.network, pretrained=True, num_classes=self.args.n_classes)
        model1.fc.weight.data = self.text_features_ems_ViT.data.float().clone().detach()
        model1 = nn.DataParallel(model1)
        self.model1 = model1.to(self.device)

        model2 = create_model(self.args.network, pretrained=True, num_classes=self.args.n_classes)
        model2.fc.weight.data = self.text_features_ems_RN.data.float().clone().detach()
        model2 = nn.DataParallel(model2)
        self.model2 = model2.to(self.device)

        self.source_loader, self.val_loader = data_helper.get_train_ems_dataloader(args, self.clip_transform)
        self.target_loader = data_helper.get_val_ems_dataloader(args, self.clip_transform)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (
        len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer1, self.scheduler1 = get_optim_and_scheduler(self.model1, args.epochs, args.learning_rate,
                                                                   args.train_all,
                                                                   nesterov=False)
        self.optimizer2, self.scheduler2 = get_optim_and_scheduler(self.model2, args.epochs, args.learning_rate_2,
                                                                   args.train_all,
                                                                   nesterov=False)
        self.current_epoch = 0
        self.distill_weight = ww1
        self.classification_weight = ww2
        self.distance_weight = ww3
        self.T = tt
        self.target_name = target_name
        print("Loss weight: distill %.4f, cls %.4f, RD %.4f. Temperature: %.4f" % (
            self.distill_weight, self.classification_weight, self.distance_weight, self.T))

    def _do_epoch(self):
        softmax = nn.Softmax(dim=1).cuda()
        criterion = nn.CrossEntropyLoss()
        cosine_sim_loss = torch.nn.CosineEmbeddingLoss()
        self.model1.train()
        self.model2.train()

        for it, ((data, data_tc, class_l), d_idx) in enumerate(self.source_loader):
            data1, data_tc1, class_l1, d_idx1 = data.to(self.device), data_tc.to(self.device), class_l.to(
                self.device), d_idx.to(self.device)
            data2, data_tc2, class_l2, d_idx2 = data.to(self.device), data_tc.to(self.device), class_l.to(
                self.device), d_idx.to(self.device)
            bs = data.shape[0]

            # Calculate features
            with torch.no_grad():
                self.clip_model_ViT.eval()
                CLIP_image_features_ViT = self.clip_model_ViT.encode_image(data1)
                CLIP_image_features_RN = self.clip_model_RN.encode_image(data_tc2)
            CLIP_image_features_ViT /= CLIP_image_features_ViT.norm(dim=-1, keepdim=True)
            CLIP_image_features_RN /= CLIP_image_features_RN.norm(dim=-1, keepdim=True)
            teacher_logits_ViT = (100.0 * CLIP_image_features_ViT @ self.text_features_ems_ViT.T).type(torch.float32)
            teacher_logits_RN = (100.0 * CLIP_image_features_RN @ self.text_features_ems_RN.T).type(torch.float32)

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            student_embedding_ViT, class_logit_ViT = self.model1(data1)
            student_embedding_RN, class_logit_RN = self.model2(data2)

            # --- classification loss
            supervised_loss1 = criterion(class_logit_ViT, class_l1)
            supervised_loss2 = criterion(class_logit_RN, class_l2)
            # --- distillation loss
            kl_loss1 = F.kl_div(F.log_softmax(class_logit_ViT / self.T, dim=1),
                                F.softmax(teacher_logits_ViT / self.T, dim=1),
                                reduction='batchmean') * self.T * self.T
            kl_loss2 = F.kl_div(F.log_softmax(class_logit_RN / self.T, dim=1),
                                F.softmax(teacher_logits_RN / self.T, dim=1),
                                reduction='batchmean') * self.T * self.T

            # --- absolute distance loss
            CLIP_text_embedding_instance_ViT = torch.zeros(student_embedding_ViT.shape[0], self.text_feature_dim).to(self.device)
            CLIP_text_embedding_instance_RN = torch.zeros(student_embedding_RN.shape[0], self.text_feature_dim).to(self.device)
            for i in range(bs):
                CLIP_text_embedding_instance_ViT[i, :] = self.text_features_ems_ViT[class_l1[i], :]
                CLIP_text_embedding_instance_RN[i, :] = self.text_features_ems_RN[class_l2[i], :]
            cosine_sim_label1 = torch.ones(student_embedding_ViT.shape[0]).to(self.device)
            cosine_sim_label2 = torch.ones(student_embedding_RN.shape[0]).to(self.device)
            text_embed_loss_sim1 = cosine_sim_loss(F.normalize(student_embedding_ViT, dim=-1),
                                                   CLIP_text_embedding_instance_ViT, cosine_sim_label1)
            text_embed_loss_sim2 = cosine_sim_loss(F.normalize(student_embedding_RN, dim=-1),
                                                   CLIP_text_embedding_instance_RN, cosine_sim_label2)

            # --- relative distance loss
            dist_teacher = torch.zeros(bs, len(self.text_anchor)).to(self.device)
            dist_student = torch.zeros(bs, len(self.text_anchor)).to(self.device)
            for pair1 in range(bs):
                tmp_anchor_feat_student = student_embedding_ViT[pair1, :]
                gt = class_l[pair1]
                tmp_anchor_feat_teacher = self.CLIP_text_features_ems_before_norm[gt]
                compare_feat = self.text_compare_teacher[gt]
                dist_teacher[pair1, :] = F.cosine_similarity(tmp_anchor_feat_teacher.repeat(len(self.text_anchor), 1), compare_feat)
                dist_student[pair1, :] = F.cosine_similarity(tmp_anchor_feat_student.repeat(len(self.text_anchor), 1), compare_feat)
            dist_teacher = softmax(dist_teacher)
            dist_student = softmax(dist_student)
            domain_feature_relation_loss = F.mse_loss(dist_student, dist_teacher) * 10.0

            class_probs_ViT = class_logit_ViT.softmax(dim=-1)
            _, cls_pred_ViT = class_probs_ViT.max(dim=1)
            class_probs_RN = class_logit_RN.softmax(dim=-1)
            _, cls_pred_RN = class_probs_RN.max(dim=1)
            loss1 = kl_loss1 * self.distill_weight \
                    + supervised_loss1 * self.classification_weight \
                    + text_embed_loss_sim1 * self.distance_weight + domain_feature_relation_loss * self.distance_weight * 0
            loss2 = kl_loss2 * self.distill_weight \
                    + supervised_loss2 * self.classification_weight \
                    + text_embed_loss_sim2 * self.distance_weight

            loss1.backward()
            self.optimizer1.step()
            loss2.backward()
            self.optimizer2.step()

            self.logger.log(it, len(self.source_loader),
                            {
                             "Loss_student1": loss1.item(), "Loss_student2": loss2.item()
                             },
                            {"class1": torch.sum(cls_pred_ViT == class_l1.data).item(),
                             "class2": torch.sum(cls_pred_RN == class_l2.data).item()
                            },
                            data.shape[0])
            del loss1, class_logit_ViT, loss2, class_logit_RN

        self.model1.eval()
        self.model2.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        class_correct = 0
        for it, ((data, data_tc, class_l), _) in enumerate(loader):
            data1, data_tc1, class_l1 = data.to(self.device), data_tc.to(self.device), class_l.to(self.device)
            data2, data_tc2, class_l2 = data.to(self.device), data_tc.to(self.device), class_l.to(self.device)
            student_embedding_ViT, student_logits_ViT = self.model1(data1)
            similarity_ViT = student_logits_ViT.softmax(dim=-1)
            student_embedding_RN, student_logits_RN = self.model2(data2)
            similarity_RN = student_logits_RN.softmax(dim=-1)
            similarity_ems = similarity_ViT + similarity_RN

            if self.args.dataset == "Terra":
                student_embedding_ViT /= student_embedding_ViT.norm(dim=-1, keepdim=True)
                student_logits_clip_1 = (100.0 * student_embedding_ViT @ self.text_features_ems_ViT.T.type(torch.float32)).type(torch.float32)
                student_embedding_RN /= student_embedding_RN.norm(dim=-1, keepdim=True)
                student_logits_clip_2 = (100.0 * student_embedding_RN @ self.text_features_ems_RN.T.type(torch.float32)).type(torch.float32)
                similarity_ems += (student_logits_clip_1.softmax(dim=-1) + student_logits_clip_2.softmax(dim=-1))

            _, cls_pred = similarity_ems.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l1.data)
        return class_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.logger.new_epoch(self.scheduler1.get_last_lr())
            self.logger.new_epoch(self.scheduler2.get_last_lr())
            self._do_epoch()
            self.scheduler1.step()
            self.scheduler2.step()
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
        return self.logger, self.model1


def train_with_sweep():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #select_txt = os.path.join(os.getcwd(), 'data', 'hp_search', args.dataset + '.txt')
    select_txt = os.path.join(os.getcwd(), 'data', 'hp_search', 'para5.txt')
    print("parameter search space: ")
    with open(select_txt, 'r') as ff:
        lines = ff.readlines()
        print(lines)

    if args.dataset == "PACS":
        args.Domain_ID = ['sketch', 'photo', 'cartoon', 'art_painting']
        args.classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        args.n_classes = 7
        args.n_domain = 4
        args.learning_rate_2 = 0.002
    elif args.dataset == "VLCS":
        args.Domain_ID = ["LABELME", "SUN", "VOC", "CALTECH"]
        args.classes = ["bird", "car", "chair", "dog", "person"]
        args.n_classes = 5
        args.n_domain = 4
        args.learning_rate_2 = 0.002
    elif args.dataset == "Terra":
        args.Domain_ID = ["location_100", "location_38", "location_43", "location_46"]
        args.classes = ["bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit", "raccoon", "squirrel"]
        args.n_classes = 10
        args.n_domain = 4
        args.learning_rate = 0.002
        args.learning_rate_2 = 0.004
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
