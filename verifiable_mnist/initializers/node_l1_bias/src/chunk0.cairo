use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 18562, sign: false });
a.append(FP16x16 { mag: 25774, sign: false });
a.append(FP16x16 { mag: 13992, sign: true });
a.append(FP16x16 { mag: 446, sign: false });
a.append(FP16x16 { mag: 213, sign: false });
a.append(FP16x16 { mag: 9209, sign: false });
a.append(FP16x16 { mag: 26513, sign: false });
a.append(FP16x16 { mag: 11846, sign: false });
a.append(FP16x16 { mag: 35279, sign: false });
a.append(FP16x16 { mag: 25997, sign: false });
}