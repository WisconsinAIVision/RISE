import os
import numpy as np
import glob
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Script to evaluate results")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--Domain_ID", default=[])
    parser.add_argument("--output_folder", default='run1', help="folder where to save results file")
    parser.add_argument("--output_file_name", default='.txt', help="results file name")

    return parser.parse_args()

def write_to_dict(filename, targetname):
    para_dict = {}
    with open(filename, 'r') as ff:
        txt = ff.readlines()
    idx1 = 0
    for t in txt:
        els = t.strip().split(',')
        els1 = els[0].strip().split(' ')[0]
        kkey = targetname + '_' + str(idx1)
        if t == '\n':
            continue
        else:
            if els[0] == 'target domain ' + targetname:
                tv = els[1].strip().split(':')[-1].strip()
                w1v = els[2].strip().split(':')[-1].strip()
                w2v = els[3].strip().split(':')[-1].strip()
                w3v = els[4].strip().split(':')[-1].strip()
                para_list = [tv, w1v, w2v, w3v]
                para_dict[kkey] = {'para': para_list,
                                   'best_val': 0.0,
                                   'correspond_test': 0.0,
                                   'best_test': 0.0,
                                   'best_epoch': 0}
            else:
                para_dict[kkey]['best_val'] = els[0].strip().split(' ')[2]
                para_dict[kkey]['correspond_test'] = els[1].strip().split(' ')[2]
                para_dict[kkey]['best_test'] = els[1].strip().split(' ')[6]
                para_dict[kkey]['best_epoch'] = els[2].strip().split(' ')[2]
                idx1 += 1
    return para_dict

def get_top_k(collect_data, top):
    dtype = [('para', list), ('best_val', float), ('correspond_test', float), ('best_test', float), ('best_epoch', int)]
    values = np.empty([len(collect_data)], dtype=dtype)
    ind = 0
    for k, v in collect_data.items():
        values[ind]['para'] = v['para']
        values[ind]['best_val'] = float(v['best_val'])
        values[ind]['correspond_test'] = float(v['correspond_test'])
        values[ind]['best_test'] = float(v['best_test'])
        values[ind]['best_epoch'] = int(v['best_epoch'])
        ind += 1
    return np.sort(values, order=['best_val', 'correspond_test'])[::-1][:top]

if __name__ == '__main__':
    args = get_args()
    if args.dataset == "PACS":
        args.Domain_ID = ['sketch', 'photo', 'cartoon', 'art_painting']
        args.n_domain = 4
    elif args.dataset == "VLCS":
        args.Domain_ID = ["LABELME", "SUN", "VOC", "CALTECH"]
        args.n_domain = 4
    elif args.dataset == "Terra":
        args.Domain_ID = ["location_100", "location_38", "location_43", "location_46"]
        args.n_domain = 4
    elif args.dataset == "Officehome":
        args.Domain_ID = ['Clipart', 'Art', 'RealWorld', 'Product']
        args.n_domain = 4
    else:
        raise NotImplementedError

    output_folder = os.path.join(os.getcwd(), 'results', args.output_folder)
    all_txt = glob.glob(output_folder + "/*.txt")
    top = 1
    i = 0
    ave_acc = 0
    for domain in args.Domain_ID:
        matching = [s for s in all_txt if domain in s]
        collect_results = write_to_dict(matching[0], domain)
        best_results = get_top_k(collect_results, top)
        print("Top {} results of {} domain:".format(top, domain))
        ave_acc += best_results[0][2]
        for i in range(top):
            print(f'Best val {best_results[i][1]:.4f}, corresponding test {best_results[i][2]:.4f} - best test: {best_results[i][3]:.4f}, best epoch: {best_results[i][4]}')
        i += 1
    print(f'Training validation results is: {(ave_acc/args.n_domain):.4f}')

