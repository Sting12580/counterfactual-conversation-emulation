export const C = {
  bg: "#F7F4EE",
  ink: "#1B1E27",
  muted: "#5D6470",
  faint: "#E5DDD1",
  line: "#CFC6B8",
  red: "#C8553D",
  teal: "#2F6F73",
  gold: "#D6A43C",
  blue: "#3D5A80",
  white: "#FFFFFF",
};

export function addBg(slide, ctx) {
  ctx.addShape(slide, {
    x: 0,
    y: 0,
    w: ctx.W,
    h: ctx.H,
    fill: C.bg,
    line: ctx.line("#00000000", 0),
  });
}

export function addKicker(slide, ctx, label, index = "01") {
  ctx.addShape(slide, {
    x: 54,
    y: 38,
    w: 32,
    h: 4,
    fill: C.red,
    line: ctx.line("#00000000", 0),
  });
  ctx.addText(slide, {
    text: `${index} / ${label}`,
    x: 96,
    y: 27,
    w: 560,
    h: 26,
    fontSize: 13,
    bold: true,
    color: C.muted,
    typeface: "Arial Unicode MS",
  });
}

export function addTitle(slide, ctx, title, subtitle) {
  ctx.addText(slide, {
    text: title,
    x: 54,
    y: 72,
    w: 1080,
    h: subtitle ? 72 : 98,
    fontSize: 32,
    bold: true,
    color: C.ink,
    typeface: "Arial Unicode MS",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  if (subtitle) {
    ctx.addText(slide, {
      text: subtitle,
      x: 56,
      y: 164,
      w: 1060,
      h: 40,
      fontSize: 16,
      color: C.muted,
      typeface: "Arial Unicode MS",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    });
  }
}

export function addFooter(slide, ctx, text = "Phase 5 / learned action embedding summary") {
  ctx.addShape(slide, {
    x: 54,
    y: 676,
    w: 1172,
    h: 1,
    fill: C.faint,
    line: ctx.line("#00000000", 0),
  });
  ctx.addText(slide, {
    text,
    x: 56,
    y: 686,
    w: 860,
    h: 20,
    fontSize: 9,
    color: "#7E766C",
    typeface: "Arial Unicode MS",
  });
}

export function addBox(slide, ctx, { x, y, w, h, fill = C.white, stroke = C.line }) {
  return ctx.addShape(slide, {
    x,
    y,
    w,
    h,
    fill,
    line: ctx.line(stroke, 1),
  });
}

export function addLabel(slide, ctx, text, x, y, w, h, opts = {}) {
  return ctx.addText(slide, {
    text,
    x,
    y,
    w,
    h,
    fontSize: opts.size ?? 15,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    typeface: "Arial Unicode MS",
    align: opts.align ?? "left",
    valign: opts.valign ?? "top",
    fill: opts.fill ?? "#00000000",
    line: ctx.line("#00000000", 0),
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function addMetric(slide, ctx, { x, y, w, label, value, note, accent = C.teal }) {
  addBox(slide, ctx, { x, y, w, h: 112, fill: "#FBFAF7", stroke: C.faint });
  ctx.addShape(slide, {
    x,
    y,
    w: 5,
    h: 112,
    fill: accent,
    line: ctx.line("#00000000", 0),
  });
  addLabel(slide, ctx, label, x + 18, y + 14, w - 28, 18, {
    size: 11,
    bold: true,
    color: C.muted,
  });
  addLabel(slide, ctx, value, x + 18, y + 36, w - 28, 34, {
    size: 25,
    bold: true,
    color: C.ink,
  });
  addLabel(slide, ctx, note, x + 18, y + 70, w - 28, 24, {
    size: 10,
    color: C.muted,
  });
}

export function addStep(slide, ctx, { x, y, w, h, title, body, accent = C.blue }) {
  addBox(slide, ctx, { x, y, w, h, fill: C.white, stroke: C.faint });
  ctx.addShape(slide, {
    x,
    y,
    w,
    h: 5,
    fill: accent,
    line: ctx.line("#00000000", 0),
  });
  addLabel(slide, ctx, title, x + 16, y + 17, w - 32, 26, {
    size: 15,
    bold: true,
  });
  addLabel(slide, ctx, body, x + 16, y + 50, w - 32, h - 68, {
    size: 12,
    color: C.muted,
  });
}

export function addArrow(slide, ctx, x, y, w, color = C.line) {
  ctx.addShape(slide, {
    x,
    y,
    w,
    h: 2,
    fill: color,
    line: ctx.line("#00000000", 0),
  });
  ctx.addShape(slide, {
    geometry: "triangle",
    x: x + w - 2,
    y: y - 5,
    w: 12,
    h: 12,
    fill: color,
    line: ctx.line("#00000000", 0),
  });
}
