import numpy as np
import torch
from torch.optim import Adam,SGD
from experiment.dataset import get_data
from models.Resnet18 import resnet18_reduced
from utils.util import compute_performance
from train_gpm import TrainLearnerGPM

def multiple_run(args):
    test_all_acc = torch.zeros(args.run_nums)
    test_all_ema_acc = torch.zeros(args.run_nums)

    accuracy_list = []
    accuracy_ema_list = []

    test_all_nme_acc = torch.zeros(args.run_nums)
    test_all_nme_ema_acc = torch.zeros(args.run_nums)

    accuracy_nme_list = []
    accuracy_nme_ema_list = []
    for run in range(args.run_nums):
        tmp_acc = []
        tmp_acc_ema = []
        tmp_acc_nme = []
        tmp_acc_nme_ema = []
        print('=' * 100)
        print(f"-----------------------------run {run} start--------------------------")
        print('=' * 100)
        data, class_num, class_per_task, task_loader, input_size = get_data(args.dataset, args.batch_size, args.n_workers,args.task_num,args.dataset_dir,args)
        args.n_classes = class_num

        model = resnet18_reduced (class_num,nf=args.nf,args=args).cuda()
        #print(model)
        total = sum([param.nelement() for param in model.parameters()]) 
        optimizer = None
        agent = TrainLearnerGPM(model, optimizer, class_num, class_per_task, input_size, args)

        for i in range(len(task_loader)):
            print(f"-----------------------------run {run} task id:{i} start training-----------------------------")
            agent.train(i, task_loader[i]['train'],task_loader)
            acc_list,acc_ema_list,acc_nme_list,acc_nme_ema_list = agent.test(i, task_loader,args.task_num)
            tmp_acc.append(acc_list.tolist())
            tmp_acc_ema.append(acc_ema_list.tolist())
            tmp_acc_nme.append(acc_nme_list.tolist())
            tmp_acc_nme_ema.append(acc_nme_ema_list.tolist())

            

        test_accuracy = acc_list.mean()
        test_all_acc[run] = test_accuracy
        accuracy_list.append(np.array(tmp_acc))

        test_accuracy_ema = acc_ema_list.mean()
        test_all_ema_acc[run] = test_accuracy_ema
        accuracy_ema_list.append(np.array(tmp_acc_ema))

        test_accuracy_nme = acc_nme_list.mean()
        test_all_nme_acc[run] = test_accuracy_nme
        accuracy_nme_list.append(np.array(tmp_acc_nme))

        test_accuracy_nme_ema = acc_nme_ema_list.mean()
        test_all_nme_ema_acc[run] = test_accuracy_nme_ema
        accuracy_nme_ema_list.append(np.array(tmp_acc_nme_ema))
        print('=' * 100)
        print("{}th run's Test result: Accuracy: {:.2f}%".format(run, test_accuracy))
        print("{}th run's Test result: EMA Accuracy: {:.2f}%".format(run, test_accuracy_ema))
        print("{}th run's Test result: NME Accuracy: {:.2f}%".format(run, test_accuracy_nme))
        print("{}th run's Test result: NME EMA Accuracy: {:.2f}%".format(run, test_accuracy_nme_ema))
        
        print('=' * 100)
        print('acc list')
        for tae in tmp_acc :
            list_tae = [round(mmm,2) for mmm in tae]
            print(list_tae)
        print('ema acc list')
        for tae in tmp_acc_ema :
            list_tae = [round(mmm,2) for mmm in tae]
            print(list_tae)
    accuracy_array = np.array(accuracy_list)
    avg_end_acc, avg_end_fgt = compute_performance(accuracy_array)
    accuracy_ema_array = np.array(accuracy_ema_list)
    avg_end_acc_ema, avg_end_fgt_ema = compute_performance(accuracy_ema_array)

    accuracy_nme_array = np.array(accuracy_nme_list)
    avg_end_acc_nme, avg_end_fgt_nme = compute_performance(accuracy_nme_array)
    accuracy_nme_ema_array = np.array(accuracy_nme_ema_list)
    avg_end_acc_ema_nme, avg_end_fgt_ema_nme = compute_performance(accuracy_nme_ema_array)
    print('=' * 100)
    print(f"total {args.run_nums}runs test acc results: {test_all_acc}")
    print('----------- Avg_End_Acc {} Avg_End_Fgt {}-----------'
          .format(avg_end_acc, avg_end_fgt))
    print(f"total {args.run_nums}runs test ema acc results: {test_all_ema_acc}")
    print('----------- Avg_End_Acc_ema {} Avg_End_Fgt_ema {}-----------'
          .format(avg_end_acc_ema, avg_end_fgt_ema))
    print(f"total {args.run_nums}runs test nme acc results: {test_all_ema_acc}")
    print('----------- Avg_End_Acc {} Avg_End_Fgt {}-----------'
          .format(avg_end_acc_nme, avg_end_fgt_nme))
    print(f"total {args.run_nums}runs test nme ema acc results: {test_all_nme_ema_acc}")
    print('----------- Avg_End_Acc_ema {} Avg_End_Fgt_ema {}-----------'
          .format(avg_end_acc_ema_nme, avg_end_fgt_ema_nme))
    print('=' * 100)
