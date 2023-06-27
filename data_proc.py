import os

from flowcontainer.extractor import extract
from flowcontainer.flows import Flow

types = ['VoIP', 'Chat', 'Email', 'Streaming', 'File']

vpn_path = 'vpn/'
avpn_path = 'avpn/'


# 判断流量类型
def get_type(file_name: str):
    if "chat" in file_name.lower():
        return 'Chat'
    elif "audio" in file_name.lower() or "voip" in file_name.lower():
        return 'VoIP'
    elif "email" in file_name.lower():
        return 'Email'
    elif "ftp" in file_name.lower() or "file" in file_name.lower():
        return 'File'
    elif "video" in file_name.lower() or "vimeo" in file_name.lower() or "spotify" in file_name.lower() \
            or "netflix" in file_name.lower():
        return 'Streaming'
    else:
        return None


# 检查单向流是否有意义
def check_valid(flow: Flow):
    prt = flow.protocol
    if str(prt) == "tcp":
        # 排除空包
        if len(flow.payload_lengths) <= 4 or max([abs(x) for x in flow.payload_lengths]) == 0:
            return False
        return True
    elif str(prt) == "udp":
        # 排除DNS和NTP解析流
        if "DNS" in flow.ext_protocol or "NTP" in flow.ext_protocol or len(flow.payload_lengths) <= 4:
            return False
        return True
    return False


# 保存单向流
flow_info = open("flow.txt", "w")

# 分析VPN包
for root, dirs, files in os.walk(vpn_path):
    for file in files:
        full_path = vpn_path + file
        t = get_type(file)
        if t is not None:
            result = extract(full_path, filter='')
            for key in result:
                value: Flow = result[key]
                if check_valid(value):
                    flow_info.write(
                        f'{value.src};{value.sport};{value.dst};{value.dport};{value.protocol};{value.payload_timestamps};'
                        f'{value.payload_lengths};{t};1' + "\n")

# 分析非VPN包
for root, dirs, files in os.walk(avpn_path):
    for file in files:
        full_path = avpn_path + file
        t = get_type(file)
        if t is not None:
            result = extract(full_path, filter='')
            for key in result:
                value: Flow = result[key]
                if check_valid(value):
                    flow_info.write(
                        f'{value.src};{value.sport};{value.dst};{value.dport};{value.protocol};{value.payload_timestamps};'
                        f'{value.payload_lengths};{t};0' + "\n")
