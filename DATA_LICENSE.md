# Data License

Copyright (c) 2026 [Yifan Chen, Jiayin Xiao, Song Wang and Yujie Wang / University of Science and Technology of China]

The contents of the `data/` directory (raw data, features, prior weights,
driving preference features, labels, and the data dictionary) are released
under the following terms.

## 1. Scope of Permission

Permission is granted, free of charge, to download and use the data **solely
for academic verification of the results of the accompanying paper**:

> "A Driving Preference-Informed Multi-Model Framework for Real-World
> Electric Vehicle Battery SOH Estimation", IEEE Transactions on
> Transportation Electrification, 2026.

Specifically, you may:
- Run the released verification script (`test_code.py`) on the released data
  and model weights;
- Compare the outputs with the pre-computed files in `results/expected_results/`;
- Reproduce the reconstructed capacity curve of Fig. 3 from the raw data of D01;
- Inspect the data structure to understand the verification workflow.

## 2. Restrictions

You may **not**:
- Use the data, in whole or in part, for any commercial purpose;
- Redistribute the data, in original or modified form, to any third party;
- Use the data to train or evaluate models for any purpose other than
  reproducing the results of the paper;
- Use the data to build a competing dataset, benchmark, or product;
- Attempt to re-identify the vehicles, devices, or users underlying the
  de-identified data.

## 3. Citation

Any use of the data must cite the paper above.

## 4. No Warranty

The data is provided "as is", without warranty of any kind, express or implied.
The authors make no representations that the data is free of errors, complete,
or fit for any particular purpose beyond the verification described above.

## 5. Termination

Any use outside the scope of this license terminates the permission granted
here immediately.

## 6. Other Uses

For any use not covered above — research collaboration, extended benchmarks,
or commercial licensing — contact the corresponding author:
Yujie Wang (wangyujie@ustc.edu.cn).

## 7. Relationship to the Code License

The code (`test_code.py`) is licensed separately under the MIT License (see
`LICENSE`). This license applies **only** to the `data/` directory.
