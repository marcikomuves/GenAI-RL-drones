from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
import os

INPUT_FILE = "runs/sac/latest_real/events.out.tfevents.1.1"
OUTPUT_DIR = "runs/sac/latest_real_new/"

REMOVE_TAGS = {
    "losses/qf1_values",
    "losses/qf2_values",
    "losses/qf1_loss",
    "losses/qf2_loss",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the input event file
acc = EventAccumulator(INPUT_FILE)
acc.Reload()

# Read all scalar events, retaining their original steps and timestamps
events_by_tag = {
    tag: acc.Scalars(tag)
    for tag in acc.Tags()["scalars"]
}

# Compute pairwise means of qf1_values and qf2_values
qf1 = {e.step: e for e in events_by_tag["losses/qf1_values"]}
qf2 = {e.step: e for e in events_by_tag["losses/qf2_values"]}

mean_events = []
for step in sorted(qf1.keys() & qf2.keys()):
    e1, e2 = qf1[step], qf2[step]
    mean_events.append((
        step,
        (e1.value + e2.value) / 2,
        max(e1.wall_time, e2.wall_time),
    ))

# Write a new event file
writer = EventFileWriter(OUTPUT_DIR)

for tag, scalar_events in events_by_tag.items():
    if tag in REMOVE_TAGS:
        continue

    for e in scalar_events:
        writer.add_event(Event(
            wall_time=e.wall_time,
            step=e.step,
            summary=Summary(value=[
                Summary.Value(tag=tag, simple_value=e.value)
            ]),
        ))

# Add the new mean scalar
for step, value, wall_time in mean_events:
    writer.add_event(Event(
        wall_time=wall_time,
        step=step,
        summary=Summary(value=[
            Summary.Value(tag="losses/qf_values", simple_value=value)
        ]),
    ))

writer.close()
print(f"Written processed TensorBoard logs to {OUTPUT_DIR}")