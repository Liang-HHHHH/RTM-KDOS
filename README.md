# RTM-KDOS
RTM-KDOS: Keypoint Detection and Operability Scoring for Multifunctional Tomato Harvesting and De-leafing Robots

Based on the RTM model, we optimized the dsconv and SimAM modules, and added a scoring module to the detection head, enabling integrated key point recognition and operability evaluation of targets. For example, for robotic operation scenarios such as tomato harvesting and de-leafing, existing models cannot judge whether the detected key points from the current viewing angle are suitable for actual operation. This newly added branch can accurately assess target operability and make up for this capability gap.

[Inference result of RTM-KDOS.mp4] This video demonstrates RTM‑KDOS accurately predicting keypoints and simultaneously outputting operability scores (not confidence). The operability score varies with the viewing angle and reaches its peak when the three keypoints lie on the same plane. RTM‑KDOS performs consistently under both front‑lighting and backlit conditions. Selected cases are partially presented in Figures 1–8.
