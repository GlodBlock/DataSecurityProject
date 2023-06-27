import numpy as np
from flowcontainer.extractor import extract
from flowcontainer.flows import Flow
import matplotlib.pyplot as plt


def check_valid(flow: Flow):
    prt = flow.protocol
    if str(prt) == "tcp":
        if len(flow.payload_lengths) <= 20 or max([abs(x) for x in flow.payload_lengths]) == 0:
            return False
        return True
    elif str(prt) == "udp":
        if "DNS" in flow.ext_protocol or "NTP" in flow.ext_protocol or len(flow.payload_lengths) <= 20:
            return False
        return True
    return False


result = extract('avpn/facebook_chat_4b.pcap', filter='')
plt.figure(figsize=(8, 8))
plt.xlim(0, 1500)
plt.ylim(0, 1500)
plt.xlabel("Normalized Timestamp")
plt.ylabel("Payload Length")
for key in result:
    value: Flow = result[key]
    if check_valid(value):
        time = np.array(value.payload_timestamps)
        size = np.array(value.payload_lengths)
        time = np.abs(time)
        size = np.abs(size)
        time -= min(time)
        time = time / (max(time) + 0.0001) * 1500
        size = np.where(size > 1500, 1500, size)
        plt.scatter(time, size, c='black')

plt.show()
