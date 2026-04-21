#!/usr/bin/env python3
"""
merge_bags.py <original_bag> <reprocessed_bag> <output_bag>

Merges two rosbags:
  - From original_bag: everything EXCEPT /image_rect and /camera/camera_info
  - From reprocessed_bag: /image_rect, /detections_reprocessed,
    /tag_measurements_base, /camera/camera_info, /tf (tag transforms)

/tf is merged from both bags - original robot frames + reprocessed tag frames.
Tag transforms already present in the original bag are stripped to avoid stale
AprilTag poses surviving into the merged output.
"""

import shutil
import sys
import re
from pathlib import Path
from rosbags.rosbag2 import Reader, Writer
from rosbags.rosbag2.writer import StoragePlugin
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from rosbags.typesys.base import TypesysError

REPROCESSED_DETECTIONS_TOPIC = '/detections_reprocessed'
REPROCESSED_MEASUREMENTS_TOPIC = '/tag_measurements_base'
REPROCESSED_TOPICS = {
    '/image_rect',
    REPROCESSED_DETECTIONS_TOPIC,
    REPROCESSED_MEASUREMENTS_TOPIC,
    '/camera/camera_info',
}
TAG_FRAME_RE = re.compile(r'^(tag_\d+|tag36h11:\d+)$')
MIN_VALID_TIMESTAMP_NS = 1


def is_tag_frame(child_frame_id):
    return bool(TAG_FRAME_RE.fullmatch(child_frame_id))


def ensure_type_registered(typestore, conn):
    if not conn.msgdef:
        return
    try:
        typestore.register(get_types_from_msg(conn.msgdef.data, conn.msgtype))
    except Exception:
        pass

def register_connection(writer, conn, typestore):
    """Register a connection, falling back to raw msgdef copy for unknown types."""
    try:
        return writer.add_connection(conn.topic, conn.msgtype, typestore=typestore)
    except TypesysError:
        pass
    if conn.msgdef and conn.digest:
        return writer.add_connection(
            conn.topic, conn.msgtype, msgdef=conn.msgdef.data, rihs01=conn.digest)
    print(f"  [SKIP] {conn.topic} ({conn.msgtype}) - unknown type, no hash")
    return None


def filter_tf_message(conn, data, typestore, *, keep_tag_frames):
    ensure_type_registered(typestore, conn)
    msg = typestore.deserialize_cdr(data, conn.msgtype)
    transforms = [
        tr for tr in msg.transforms
        if is_tag_frame(tr.child_frame_id) == keep_tag_frames
    ]
    if not transforms:
        return None
    msg.transforms = transforms
    return typestore.serialize_cdr(msg, conn.msgtype)


def header_stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def message_timestamp_ns(conn, data, typestore):
    ensure_type_registered(typestore, conn)
    msg = typestore.deserialize_cdr(data, conn.msgtype)

    if hasattr(msg, "header"):
        stamp_ns = header_stamp_to_ns(msg.header.stamp)
        return stamp_ns if stamp_ns >= MIN_VALID_TIMESTAMP_NS else None

    if conn.topic in {"/tf", "/tf_static"} and getattr(msg, "transforms", None):
        stamps = [
            header_stamp_to_ns(tr.header.stamp)
            for tr in msg.transforms
            if header_stamp_to_ns(tr.header.stamp) >= MIN_VALID_TIMESTAMP_NS
        ]
        if stamps:
            return min(stamps)

    return None


def reprocessed_message_key(conn, data, typestore):
    if conn.topic not in REPROCESSED_TOPICS:
        return None

    ensure_type_registered(typestore, conn)
    msg = typestore.deserialize_cdr(data, conn.msgtype)
    stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

    if conn.topic == REPROCESSED_DETECTIONS_TOPIC:
        return (conn.topic, stamp, tuple(det.id for det in msg.detections))

    if conn.topic == REPROCESSED_MEASUREMENTS_TOPIC:
        ids = ()
        for channel in getattr(msg, 'channels', []):
            if channel.name == 'id':
                ids = tuple(int(round(value)) for value in channel.values)
                break
        return (conn.topic, stamp, ids)

    return (conn.topic, stamp)


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <original_bag> <reprocessed_bag> <output_bag>")
        sys.exit(1)

    original_path    = Path(sys.argv[1])
    reprocessed_path = Path(sys.argv[2])
    output_path      = Path(sys.argv[3])

    if output_path.exists():
        print(f"[WARN] Output '{output_path}' exists - removing.")
        shutil.rmtree(output_path)

    typestore = get_typestore(Stores.ROS2_JAZZY)

    with Reader(original_path) as orig, \
         Reader(reprocessed_path) as reprocessed, \
         Writer(output_path, version=9, storage_plugin=StoragePlugin.MCAP) as writer:

        # Register connections (track by topic to avoid duplicates)
        topic_conn_map = {}  # topic -> output connection

        orig_conn_map = {}
        for conn in orig.connections:
            if conn.topic in REPROCESSED_TOPICS:
                continue
            if conn.topic not in topic_conn_map:
                out_conn = register_connection(writer, conn, typestore)
                if out_conn:
                    topic_conn_map[conn.topic] = out_conn
            if conn.topic in topic_conn_map:
                orig_conn_map[conn.id] = topic_conn_map[conn.topic]

        reprocessed_conn_map = {}
        for conn in reprocessed.connections:
            if conn.topic not in topic_conn_map:
                out_conn = register_connection(writer, conn, typestore)
                if out_conn:
                    topic_conn_map[conn.topic] = out_conn
            if conn.topic in topic_conn_map:
                reprocessed_conn_map[conn.id] = topic_conn_map[conn.topic]

        # Compute timestamp offset to align reprocessed bag with original
        orig_start = next(orig.messages())[1]
        repr_start = next(reprocessed.messages())[1]
        ts_offset = orig_start - repr_start
        print(f"Timestamp offset: {ts_offset / 1e9:.1f}s "
              f"({'adding' if ts_offset > 0 else 'subtracting'} "
              f"{abs(ts_offset) / 1e9:.1f}s to reprocessed timestamps)")

        # Write original messages (excluding replaced topics)
        print(f"Writing from original bag: {original_path.name}")
        orig_counts = {}
        orig_filtered_tf = 0
        for conn, timestamp, data in orig.messages():
            if conn.topic in REPROCESSED_TOPICS:
                continue
            if conn.id not in orig_conn_map:
                continue
            if conn.topic in {'/tf', '/tf_static'}:
                filtered = filter_tf_message(conn, data, typestore, keep_tag_frames=False)
                if filtered is None:
                    orig_filtered_tf += 1
                    continue
                data = filtered
            writer.write(orig_conn_map[conn.id], timestamp, data)
            orig_counts[conn.topic] = orig_counts.get(conn.topic, 0) + 1

        # Write reprocessed messages (with timestamp offset applied)
        print(f"Writing from reprocessed bag: {reprocessed_path.name}")
        repr_counts = {}
        repr_skipped_duplicates = 0
        repr_filtered_tf = 0
        repr_header_timestamp_used = {}
        repr_fallback_timestamp_used = {}
        seen_reprocessed = set()
        for conn, timestamp, data in reprocessed.messages():
            if conn.id not in reprocessed_conn_map:
                continue
            key = reprocessed_message_key(conn, data, typestore)
            if key is not None:
                if key in seen_reprocessed:
                    repr_skipped_duplicates += 1
                    continue
                seen_reprocessed.add(key)
            if conn.topic == '/tf':
                filtered = filter_tf_message(conn, data, typestore, keep_tag_frames=True)
                if filtered is None:
                    repr_filtered_tf += 1
                    continue
                data = filtered
            elif conn.topic == '/tf_static':
                filtered = filter_tf_message(conn, data, typestore, keep_tag_frames=False)
                if filtered is None:
                    repr_filtered_tf += 1
                    continue
                data = filtered
            write_timestamp = message_timestamp_ns(conn, data, typestore)
            if write_timestamp is None:
                write_timestamp = timestamp + ts_offset
                repr_fallback_timestamp_used[conn.topic] = repr_fallback_timestamp_used.get(conn.topic, 0) + 1
            else:
                repr_header_timestamp_used[conn.topic] = repr_header_timestamp_used.get(conn.topic, 0) + 1
            writer.write(reprocessed_conn_map[conn.id], write_timestamp, data)
            repr_counts[conn.topic] = repr_counts.get(conn.topic, 0) + 1

    print("\n=== Merge complete ===")
    print(f"Output: {output_path}")
    print("\nFrom original bag:")
    for topic, count in sorted(orig_counts.items()):
        print(f"  {topic}: {count}")
    print(f"  filtered original tf/tf_static messages: {orig_filtered_tf}")
    print("\nFrom reprocessed bag:")
    for topic, count in sorted(repr_counts.items()):
        print(f"  {topic}: {count}")
    print(f"  skipped duplicate reprocessed messages: {repr_skipped_duplicates}")
    print(f"  filtered reprocessed tf/tf_static messages: {repr_filtered_tf}")
    print("  reprocessed timing source:")
    for topic in sorted(repr_counts):
        header_used = repr_header_timestamp_used.get(topic, 0)
        fallback_used = repr_fallback_timestamp_used.get(topic, 0)
        print(f"    {topic}: header={header_used} fallback={fallback_used}")

if __name__ == '__main__':
    main()
