use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 50698, sign: true });
a.append(FP16x16 { mag: 37113, sign: false });
a.append(FP16x16 { mag: 16936, sign: false });
a.append(FP16x16 { mag: 19931, sign: true });
a.append(FP16x16 { mag: 24653, sign: true });
a.append(FP16x16 { mag: 76811, sign: false });
a.append(FP16x16 { mag: 21603, sign: true });
a.append(FP16x16 { mag: 20170, sign: false });
a.append(FP16x16 { mag: 40308, sign: true });
a.append(FP16x16 { mag: 15733, sign: true });
}