import { C, addBg, addBox, addFooter, addKicker, addLabel, addTitle } from "./common.mjs";

function tableRow(slide, ctx, y, cells, opts = {}) {
  const xs = [76, 365, 520, 655, 805, 940, 1075];
  const ws = [270, 110, 95, 115, 100, 105, 120];
  if (opts.fill) {
    ctx.addShape(slide, {
      x: 62,
      y: y - 8,
      w: 1135,
      h: 54,
      fill: opts.fill,
      line: ctx.line("#00000000", 0),
    });
  }
  cells.forEach((cell, i) => {
    addLabel(slide, ctx, cell, xs[i], y, ws[i], 30, {
      size: opts.header ? 11 : 13,
      bold: opts.header || opts.bold,
      color: opts.header ? C.muted : C.ink,
      align: i === 0 ? "left" : "right",
    });
  });
}

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addKicker(slide, ctx, "RESULTS", "04");
  addTitle(
    slide,
    ctx,
    "concat-PCA 带来小幅但一致的 point-estimation 提升；CI 仍然不覆盖 truth",
    "当前结论是：BGE-M3 learned concat-PCA + MIPS 是最好的点估计方案，但 plain bootstrap CI 仍然偏低。"
  );

  addBox(slide, ctx, { x: 55, y: 218, w: 1165, h: 275, fill: "#FFFFFF", stroke: C.faint });
  tableRow(slide, ctx, 242, ["Setup", "Estimator", "V_hat", "RelBias", "RMSE", "Dir%", "95% CI"], {
    header: true,
  });
  ctx.addShape(slide, { x: 70, y: 278, w: 1130, h: 1, fill: C.faint, line: ctx.line("#00000000", 0) });
  tableRow(slide, ctx, 300, ["BGE-M3 baseline", "MIPS", "0.7659", "-8.97%", "0.0705", "100%", "[0.750, 0.793]"]);
  tableRow(slide, ctx, 348, ["Pure learned 128d", "MIPS", "0.7193", "-14.50%", "0.1134", "100%", "—"]);
  tableRow(slide, ctx, 396, ["BGE-M3 learned concat-PCA", "MIPS", "0.7711", "-8.34%", "0.0674", "100%", "[0.751, 0.799]"], {
    bold: true,
    fill: "#FFF2CC",
  });

  addBox(slide, ctx, { x: 92, y: 535, w: 315, h: 92, fill: "#FBFAF7", stroke: C.faint });
  addLabel(slide, ctx, "Improvement vs BGE baseline", 112, 552, 260, 22, {
    size: 14,
    bold: true,
    color: C.teal,
  });
  addLabel(slide, ctx, "V_hat: 0.7659 → 0.7711\nRMSE: 0.0705 → 0.0674", 112, 582, 250, 34, {
    size: 12,
    color: C.muted,
  });

  addBox(slide, ctx, { x: 482, y: 535, w: 315, h: 92, fill: "#FBFAF7", stroke: C.faint });
  addLabel(slide, ctx, "Still underestimates value", 502, 552, 260, 22, {
    size: 14,
    bold: true,
    color: C.red,
  });
  addLabel(slide, ctx, "V_true_agent = 0.8413\nbest V_hat = 0.7711", 502, 582, 250, 34, {
    size: 12,
    color: C.muted,
  });

  addBox(slide, ctx, { x: 872, y: 535, w: 315, h: 92, fill: "#FBFAF7", stroke: C.faint });
  addLabel(slide, ctx, "Plain CI misses truth", 892, 552, 260, 22, {
    size: 14,
    bold: true,
    color: C.red,
  });
  addLabel(slide, ctx, "CI upper = 0.7994\ntruth gap ≈ 0.0419", 892, 582, 250, 34, {
    size: 12,
    color: C.muted,
  });

  addFooter(slide, ctx, "Mixed/Sonnet46 Phase 5 full bootstrap results; n=397, n_boot=100");
  return slide;
}
