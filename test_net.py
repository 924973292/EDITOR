import os
from config import cfg
import argparse
from data import make_dataloader
from modeling import make_model
from engine.processor import do_inference
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionReID Testing")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # Which feature to test
    # 0->ALL features 1->Original_r 2->Original_f 3->LRU_r 4->LRU_f 5->SRM_r 6->SRM_f
    # TEST.FEAT = 0
    parser.add_argument("--fea_cft", default=0, help="Feature choose to be tested", type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.FEAT = args.fea_cft
    cfg.freeze()
    if cfg.TEST.FEAT == 0:
        print('All features used in test')
    elif cfg.TEST.FEAT == 1:
        print('Original_r used in test')
    elif cfg.TEST.FEAT == 2:
        print('Original_f used in test')
    elif cfg.TEST.FEAT == 3:
        print('LRU_r used in test')
    elif cfg.TEST.FEAT == 4:
        print('LRU_f used in test')
    elif cfg.TEST.FEAT == 5:
        print('SRM_r used in test')
    elif cfg.TEST.FEAT == 6:
        print('SRM_f used in test')
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("FusionReID", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    file = cfg.OUTPUT_DIR.replace('.', '')
    model.load_param('/15127306268/wyh/UIS' + file + '/FusionReID_180.pth')

    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(
                cfg)
            rank_1, rank5 = do_inference(cfg,
                                         model,
                                         val_loader,
                                         num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum() / 10.0, all_rank_5.sum() / 10.0))
    else:
        do_inference(cfg,
                     model,
                     val_loader,
                     num_query)
