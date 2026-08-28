---
title: Vision-Based Lane Inference
emoji: 🛣️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Vision-Based Lane Inference for Unstructured and Structured Indian Roads

Harshwardhan Mukund Mohadikar · Shreyas Bhat K
Supervised by Dr. Ashok Yemineni — BITS Pilani, BSc Computer Science

Most lane systems detect painted markings. On an unmarked Indian road there is
no paint, yet the road still carries a lane structure drivers agree on. This
system infers it: a segmentation network trained on the Indian Driving Dataset
produces a drivable-surface mask, the vanishing point of the road boundaries
gives the camera pitch, the ground plane is rectified to a metric bird's-eye
view, and the carriageway is divided by the IRC design lane width to give a lane
*count* rather than an arbitrary subdivision.

* `/project` — the technical report
* `/demo` — upload an image, process a clip, or use a live camera

Running on CPU. The segmentation network is 3.33 M parameters and takes about
75 ms per frame on two threads.
