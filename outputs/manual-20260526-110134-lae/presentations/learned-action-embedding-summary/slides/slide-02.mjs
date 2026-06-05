import { C, addArrow, addBg, addBox, addFooter, addKicker, addLabel, addMetric, addStep, addTitle } from "./common.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addKicker(slide, ctx, "PURE LEARNED VERSION", "02");
  addTitle(
    slide,
    ctx,
    "第一版 pure learned 128 维把 BGE-M3 替换掉，结果反而更差",
    "这一版冻结 BGE-M3，只用 clinician logged tuples 学 projection；训练后再把同一个 action projector 应用到 agent action。"
  );

  addStep(slide, ctx, {
    x: 60,
    y: 220,
    w: 205,
    h: 120,
    title: "BGE-M3 frozen",
    body: "phi_x, phi_a_clinician, phi_a_agent are fixed 1024-d vectors.",
    accent: C.blue,
  });
  addArrow(slide, ctx, 276, 278, 58);
  addStep(slide, ctx, {
    x: 350,
    y: 220,
    w: 235,
    h: 120,
    title: "Train projection",
    body: "Use only (phi_x, phi_a_clinician) -> y_score. Agent reward is not used.",
    accent: C.teal,
  });
  addArrow(slide, ctx, 596, 278, 58);
  addStep(slide, ctx, {
    x: 670,
    y: 220,
    w: 220,
    h: 120,
    title: "Replace features",
    body: "z_x = f(phi_x), z_a = g(phi_a). Feature becomes [z_x; z_a; z_x*z_a].",
    accent: C.gold,
  });
  addArrow(slide, ctx, 900, 278, 58);
  addStep(slide, ctx, {
    x: 975,
    y: 220,
    w: 210,
    h: 120,
    title: "Run OPE",
    body: "DM, MIPS, OffCEM are re-run on 384-d estimator features.",
    accent: C.red,
  });

  addBox(slide, ctx, { x: 70, y: 392, w: 505, h: 150, fill: "#FBFAF7", stroke: C.faint });
  addLabel(slide, ctx, "Feature construction in pure learned mode", 94, 415, 430, 22, {
    size: 17,
    bold: true,
  });
  addLabel(
    slide,
    ctx,
    "clinician feature = [z_x ; z_a_clinician ; z_x * z_a_clinician]\nagent feature     = [z_x ; z_a_agent ; z_x * z_a_agent]\n\nlearned_dim = 128  →  estimator feature dim = 128 × 3 = 384",
    94,
    452,
    455,
    76,
    { size: 13, color: C.muted }
  );

  addMetric(slide, ctx, {
    x: 640,
    y: 392,
    w: 175,
    label: "MIPS V_hat",
    value: "0.7193",
    note: "truth = 0.8413",
    accent: C.red,
  });
  addMetric(slide, ctx, {
    x: 835,
    y: 392,
    w: 175,
    label: "REL BIAS",
    value: "-14.50%",
    note: "worse than BGE-M3 baseline",
    accent: C.red,
  });
  addMetric(slide, ctx, {
    x: 1030,
    y: 392,
    w: 175,
    label: "RMSE",
    value: "0.1134",
    note: "larger error",
    accent: C.red,
  });

  addLabel(
    slide,
    ctx,
    "Interpretation: the projection learns reward-informed structure, but full replacement likely overfits and suppresses BGE-M3's original semantic/OPE signal.",
    90,
    585,
    1090,
    42,
    { size: 18, bold: true, color: C.ink, align: "center" }
  );
  addFooter(slide, ctx);
  return slide;
}
