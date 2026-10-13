"""Offline, streaming, bidirectional flow extraction. Never sends network traffic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import dpkt
from .core import CANONICAL

FIELDS = ['flow_id', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'] + CANONICAL

def extract(source, destination, active_timeout=60., idle_timeout=15., max_open=100000):
    active, previous, skipped, packets, flows = {}, None, 0, 0, 0
    def row(flow):
        duration = flow['last'] - flow['start']
        count = sum(flow['packets'])
        size = sum(flow['bytes'])
        identity = '|'.join(map(str, flow['first'])) + '|' + repr(flow['start'])
        from datetime import datetime, timezone
        return {'flow_id': hashlib.sha256(identity.encode()).hexdigest()[:24],
                'timestamp': datetime.fromtimestamp(flow['start'], timezone.utc).isoformat(),
                'src_ip': flow['first'][0], 'dst_ip': flow['first'][2],
                'src_port': flow['first'][1], 'dst_port': flow['first'][3], 'protocol': flow['first'][4],
                'duration_s': duration, 'src_packets': flow['packets'][0], 'dst_packets': flow['packets'][1],
                'src_ip_bytes': flow['bytes'][0], 'dst_ip_bytes': flow['bytes'][1],
                'mean_ip_len': size/count, 'packets_per_s': count/max(duration, 1e-6),
                'bytes_per_s': size/max(duration, 1e-6)}
    with open(destination, 'w', newline='') as fp, open(source, 'rb') as capture:
        try:
            reader = dpkt.pcap.Reader(capture)
        except ValueError:
            capture.seek(0)
            reader = dpkt.pcapng.Reader(capture)
        linktype = reader.datalink()
        if linktype not in [1, 101, 113, 228, 229]:
            raise ValueError(f'Unsupported link type {linktype}; supported: Ethernet, raw IP and Linux cooked v1.')
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        for stamp, payload in reader:
            packets += 1
            now = float(stamp)
            if previous is not None and now < previous:
                raise ValueError('PCAP must be timestamp-sorted; do not reorder captures silently.')
            previous = now
            # Flush periodically; matching flow is also expired before accepting a packet.
            if packets % 1000 == 0:
                expired = [k for k, v in active.items() if now-v['last'] > idle_timeout or now-v['start'] >= active_timeout]
                for key in expired:
                    writer.writerow(row(active.pop(key)))
                    flows += 1
            try:
                if linktype == 1:
                    layer = dpkt.ethernet.Ethernet(payload).data
                elif linktype == 113:
                    layer = dpkt.sll.SLL(payload).data
                else:
                    layer = dpkt.ip.IP(payload) if payload[0] >> 4 == 4 else dpkt.ip6.IP6(payload)
            except (dpkt.UnpackError, ValueError, IndexError):
                raise ValueError(f'Malformed/truncated frame at packet {packets}.')
            if not isinstance(layer, (dpkt.ip.IP, dpkt.ip6.IP6)) or not isinstance(layer.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                skipped += 1
                continue
            # Fragmented IPv4/IPv6 frames are excluded: no reassembly is performed.
            if isinstance(layer, dpkt.ip.IP) and (layer.offset != 0 or layer.mf):
                skipped += 1
                continue
            if isinstance(layer, dpkt.ip6.IP6) and 44 in layer.extension_hdrs:
                skipped += 1
                continue
            import socket
            family = socket.AF_INET if isinstance(layer, dpkt.ip.IP) else socket.AF_INET6
            src, dst = socket.inet_ntop(family, layer.src), socket.inet_ntop(family, layer.dst)
            transport = layer.data
            protocol = 'tcp' if isinstance(transport, dpkt.tcp.TCP) else 'udp'
            forward = (src, int(transport.sport), dst, int(transport.dport), protocol)
            reverse = (dst, int(transport.dport), src, int(transport.sport), protocol)
            key = min(forward, reverse)
            current = active.get(key)
            if current and (now-current['last'] > idle_timeout or now-current['start'] >= active_timeout):
                writer.writerow(row(active.pop(key)))
                flows += 1
                current = None
            if current is None:
                if len(active) >= max_open:
                    raise ValueError('Open-flow cap exceeded. Use smaller captures or raise max-open explicitly.')
                current = {'first': forward, 'start': now, 'last': now, 'packets': [0, 0], 'bytes': [0, 0]}
                active[key] = current
            direction = 0 if forward == current['first'] else 1
            ip_len = int(layer.len) if isinstance(layer, dpkt.ip.IP) else 40+int(layer.plen)
            if len(bytes(layer)) < ip_len:
                raise ValueError('Truncated IP frame detected. Capture full packets for this schema.')
            current['last'] = now
            current['packets'][direction] += 1
            current['bytes'][direction] += ip_len
            # FIN/RST are retained in the window; no TCP connection state inference.
        for current in active.values():
            writer.writerow(row(current))
            flows += 1
    digest = hashlib.sha256()
    with open(source, 'rb') as fp:
        for chunk in iter(lambda: fp.read(1024*1024), b''):
            digest.update(chunk)
    return {'schema': 'canonical-pcap-v1', 'packets_seen': packets, 'flows_written': flows,
            'skipped_non_tcp_udp_or_fragmented': skipped, 'active_timeout_s': active_timeout,
            'idle_timeout_s': idle_timeout, 'source_sha256': digest.hexdigest()}

def main():
    p = argparse.ArgumentParser(description='Extract canonical offline flow CSV from authorized PCAP')
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-open', type=int, default=100000)
    args = p.parse_args()
    destination = Path(args.out)
    partial = destination.with_suffix(destination.suffix+'.partial')
    try:
        result = extract(args.input, partial, max_open=args.max_open)
        partial.replace(destination)
        destination.with_suffix('.manifest.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        partial.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
