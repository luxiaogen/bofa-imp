import socket
import os


def init_wandb(args, log_dir):
    if not args.get("use_wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed. Run `pip install wandb` or remove `--use_wandb`.") from exc

    wandb.init(project=args["project_name"], config=args, dir=log_dir)
    return wandb.run


def log_wandb(metrics):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(metrics)


def get_local_ip():
    try:
        # 创建一个UDP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到一个外部地址（这里使用Google的DNS服务器）
        s.connect(("8.8.8.8", 80))
        # 获取本地IP地址
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return local_ip


def get_root_path():
    host_name = socket.gethostname()
    # 添加默认路径，避免报错
    path = os.getcwd()  # 使用当前工作目录作为默认路径

    if host_name == "LAMDA1-GPU-K80-145":
        path = "/home/yehj"
    elif host_name == "lan-Super-Server":
        path = "/home/lan"
    elif host_name == "LAMDA1-GPU2":
        path = "/user/lan"
    elif host_name == "amax":
        local_ip = get_local_ip()
        if local_ip == "210.28.134.233":
            path = "/data/lil"
        else:
            path = "/user/lil"
    elif host_name == "LAMDA1-GPU-A100-93":
        path = "/home/v-lil/share/result"
    elif host_name == "lamda3-desktop":
        path = "/home/lil"
    elif host_name == "amax-Super-Server":
        path = "/user/lil/ss"
    # else:
    #     raise NotImplementedError("No that server:{}".format(host_name))
    return host_name, path


def get_data_root_path():
    host_name = socket.gethostname()
    # 添加默认路径，避免报错
    path = os.getcwd()  # 使用当前工作目录作为默认路径

    if host_name == "LAMDA1-GPU-K80-145":
        path = "/home/yehj"
    elif host_name == "lan-Super-Server":
        path = "/home/lan/project_share"
    elif host_name == "LAMDA1-GPU2":
        path = "/user/lan"
    elif host_name == "amax":
        local_ip = get_local_ip()
        if local_ip == "210.28.134.233":
            path = "/data/lil"
        else:
            path = "/user/lil/data"
    elif host_name == "LAMDA1-GPU-A100-93":
        path = "/home/v-lil/share"
    elif host_name == "lamda3-desktop":
        path = "/home/lil/share"
    elif host_name == "amax-Super-Server":
        path = "/user/lil/share"
    # else:
    #     raise NotImplementedError("No that server:{}".format(host_name))
    return path


def get_result_path():
    HOST_NAME, ROOT_PATH = get_root_path()
    path = os.path.join(ROOT_PATH, "result")
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def make_new_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)