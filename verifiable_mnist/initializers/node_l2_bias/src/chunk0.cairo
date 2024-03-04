use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 13590, sign: false });
a.append(FP16x16 { mag: 103, sign: false });
a.append(FP16x16 { mag: 3021, sign: false });
a.append(FP16x16 { mag: 14084, sign: false });
a.append(FP16x16 { mag: 2859, sign: false });
a.append(FP16x16 { mag: 29767, sign: false });
a.append(FP16x16 { mag: 34025, sign: true });
a.append(FP16x16 { mag: 39679, sign: false });
a.append(FP16x16 { mag: 64578, sign: true });
a.append(FP16x16 { mag: 15941, sign: true });
}