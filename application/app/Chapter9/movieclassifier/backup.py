from shutil import copyfile
import time
import os

cur_dir = os.path.dirname(__file__)
timestr = time.strftime("%Y%m%d-%H%M%S")

orig_path = os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl')
backup_path = os.path.join(cur_dir, 'pkl_objects', 'backup_classifier_%s.pkl' % timestr)
copyfile(orig_path, backup_path)
