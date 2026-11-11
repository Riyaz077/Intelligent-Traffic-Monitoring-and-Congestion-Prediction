# Limitations and Responsible Use

- A pretrained general-purpose detector may miss, misclassify or double-count vehicles in crowded scenes, at night, in rain or under occlusion.
- Sampled-frame counts are not traffic flow, occupancy, queue length or vehicle throughput.
- Congestion thresholds are heuristic and must be calibrated to a particular camera and road geometry.
- No future-time congestion prediction model is included in this starter scaffold.
- Speed estimation needs calibrated camera geometry and reliable tracking; it is not implemented here.
- Do not use the output for enforcement, emergency routing or safety-critical decisions.
- Process only footage you are authorized to use; consider blurring faces and license plates, limiting retention and following applicable privacy rules.
