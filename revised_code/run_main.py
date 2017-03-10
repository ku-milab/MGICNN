import numpy as np

#from revised_code import preprocess
from revised_code import train_model


# def run_preprocess(dataset, irs_dataset, extract_size, mean_size, patch_size, is_inception):
#     preprocess.save_orig_data()
#     preprocess.train_IRS()
#     preprocess.set_IRS(irs_dataset=irs_dataset)
#     preprocess.get_mean_var(dataset=dataset, irs_dataset=irs_dataset)
#     preprocess.get_patch_info(dataset=dataset)
#     preprocess.get_extract_idx(dataset=dataset, extract_size=extract_size)


# def run_etc(dataset, irs_dataset):
#     model_exec = train_model.model_execute(dataset=dataset, irs_dataset=irs_dataset)
#     model_exec.extract_patch_original_CNN()


def run_train(dataset, irs_dataset):
    model_def = train_model.model_def()
    model_exec = train_model.model_execute(dataset=dataset, irs_dataset=irs_dataset)

    cross_entropy, softmax, layers, data_node, label_node = model_def.original_CNN(train=True)
    model_exec.train_original_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
                                  label_node=label_node)
    cross_entropy, softmax, layers, data_node, label_node = model_def.original_CNN(train=False)
    model_exec.test_original_CNN(softmax=softmax, data_node=data_node, dataset="t", model_epoch=str(10))


def run_code_test():
    model_exec = train_model.model_execute(dataset=dataset, irs_dataset=irs_dataset)
    model_exec.code_test()
    # model_def = train_model.model_def()
    #
    # cross_entropy, softmax, layers, data_node, label_node = model_def.original_CNN(train=True)
    # model_exec.train_original_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
    #                               label_node=label_node)


#if __name__ == "__main__":
#    dataset = "output/"
#    irs_dataset = "data/"
    #extract_size = 4000000 * np.array([0.5, 0.125, 0.125, 0.125, 0.125])
    #mean_size = 0
    #patch_size = 33
#    is_inception = False
    #run_preprocess(dataset, irs_dataset, extract_size, mean_size, patch_size, is_inception)
    # run_etc(dataset, irs_dataset)
#    run_train(dataset, irs_dataset)
    # run_code_test()
