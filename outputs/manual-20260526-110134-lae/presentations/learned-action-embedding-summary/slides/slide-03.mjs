import { C, addArrow, addBg, addBox, addFooter, addKicker, addLabel, addStep, addTitle } from "./common.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addKicker(slide, ctx, "CONSERVATIVE REVISION", "03");
  addTitle(
    slide,
    ctx,
    "concat-PCA 的改动：保留 BGE 主信号，只让 learned embedding 做补充",
    "我们从 replace 改为 concat，把 PCA(BGE) 和 learned projection 拼在一起，降低破坏原始语义几何的风险。"
  );

  addBox(slide, ctx, { x: 70, y: 220, w: 500, h: 305, fill: "#FFFFFF", stroke: C.faint });
  addLabel(slide, ctx, "Before: replace", 100, 242, 300, 28, { size: 21, bold: true, color: C.red });
  addLabel(slide, ctx, "z = learned_projection(BGE)", 100, 292, 360, 26, { size: 17, bold: true });
  addArrow(slide, ctx, 210, 336, 130, C.red);
  addLabel(slide, ctx, "feature = [z_x ; z_a ; z_x * z_a]", 100, 385, 370, 28, { size: 16 });
  addLabel(
    slide,
    ctx,
    "Problem: learned representation完全替换 BGE-M3，容易压掉原本有用的 semantic geometry。",
    100,
    448,
    390,
    52,
    { size: 13, color: C.muted }
  );

  addBox(slide, ctx, { x: 710, y: 220, w: 500, h: 305, fill: "#FFFFFF", stroke: C.faint });
  addLabel(slide, ctx, "After: concat-PCA", 740, 242, 330, 28, { size: 21, bold: true, color: C.teal });
  addLabel(slide, ctx, "z = [PCA(BGE) ; learned_projection(BGE)]", 740, 292, 410, 30, {
    size: 16,
    bold: true,
  });
  addArrow(slide, ctx, 850, 336, 130, C.teal);
  addLabel(slide, ctx, "feature = [z_x ; z_a ; z_x * z_a]", 740, 385, 370, 28, { size: 16 });
  addLabel(
    slide,
    ctx,
    "Benefit: PCA(BGE) keeps original BGE-M3 geometry; learned projection contributes reward-informed correction.",
    740,
    448,
    390,
    52,
    { size: 13, color: C.muted }
  );

  addStep(slide, ctx, {
    x: 210,
    y: 565,
    w: 235,
    h: 70,
    title: "PCA dim = 128",
    body: "Unsupervised compression of BGE-M3 1024-d signal.",
    accent: C.blue,
  });
  addStep(slide, ctx, {
    x: 520,
    y: 565,
    w: 235,
    h: 70,
    title: "learned dim = 64",
    body: "Reward-informed low-dimensional correction.",
    accent: C.teal,
  });
  addStep(slide, ctx, {
    x: 830,
    y: 565,
    w: 235,
    h: 70,
    title: "final dim = 192",
    body: "Estimator feature dim = 192 × 3 = 576.",
    accent: C.gold,
  });

  addFooter(slide, ctx);
  return slide;
}
