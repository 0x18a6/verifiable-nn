use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 16739, sign: false });
a.append(FP16x16 { mag: 16492, sign: false });
a.append(FP16x16 { mag: 44838, sign: false });
a.append(FP16x16 { mag: 57908, sign: false });
a.append(FP16x16 { mag: 10300, sign: true });
a.append(FP16x16 { mag: 8210, sign: true });
a.append(FP16x16 { mag: 4911, sign: false });
a.append(FP16x16 { mag: 2412, sign: false });
a.append(FP16x16 { mag: 20308, sign: false });
a.append(FP16x16 { mag: 5360, sign: false });
}