import os
import shutil
import glob

def file_backup(args):
    if args.wandb:
        experimentdir = f"./code/{args.name}_{args.train_version}"
        args.codepath = experimentdir
    else:
        experimentdir = f"./code/{args.name}_test"

    if not os.path.exists("./code"):
        os.mkdir("./code")

    if os.path.exists(experimentdir):
        print(experimentdir + ' : exists. overwrite it.')
        shutil.rmtree(experimentdir)
        os.mkdir(experimentdir)
    else:
        os.mkdir(experimentdir)

    shutil.copytree(f"./zk_utils", os.path.join(experimentdir, 'zk_utils'))
    # shutil.copytree(f"./../bash_files", os.path.join(experimentdir, 'bash_files'))
    shutil.copytree(f"./scripts", os.path.join(experimentdir, 'scripts'))
    shutil.copytree(f"./models", os.path.join(experimentdir, 'models'))

    # search bash files and save it 
    pathname = "./*.py"
    files = glob.glob(pathname, recursive=True)

    for file in files:
        dest_fpath = os.path.join(experimentdir, file.split("/")[-1])
        try:
            shutil.copy(file, dest_fpath)
        except IOError as io_err:
            os.makedirs(os.path.dirname(dest_fpath))
            shutil.copy(file, dest_fpath)
    