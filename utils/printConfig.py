#
# 输出当前模型的配置信息
#

def configInfo(paras):
    """
    根据模型运行的配置信息进行输出
    """

    print("\n===========================")
    for k,v in paras.items():
        print("%s = %s" %(k, v))
    print("===========================\n")
    