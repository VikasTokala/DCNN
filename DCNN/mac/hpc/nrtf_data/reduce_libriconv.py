import mac
import os 

if __name__ == '__main__':

    home = os.path.expanduser('~')
    dataset_dir = home + '/mac/datasets/libriconv_8ch/'
    output_dir = home + '/mac/datasets/libriconv_4ch/'
    mac.nrtf.reduce_libriconv_dataset(dataset_dir, output_dir, 4)