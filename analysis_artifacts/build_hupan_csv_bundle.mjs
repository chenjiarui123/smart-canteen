import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const sourceRoot = "/private/tmp/smart_canteen_update_20260824_build";
const outputRoot = path.resolve("outputs/data_update_hupan_csv_20260824");
const previewRoot = "/private/tmp/smart_canteen_update_hupan_csv_previews";

const files = [
  {
    source: "weather.csv",
    output: "weather_20260611_20260824.csv",
    sheetName: "天气",
    expectedRows: 76,
    headers: [
      "日期",
      "气温（℃）",
      "相对湿度（%）",
      "降水量（毫米）",
      "10米风速（公里/小时）",
    ],
  },
  {
    source: "calendar.csv",
    output: "calendar_20260611_20260824.csv",
    sheetName: "校历",
    expectedRows: 76,
    headers: [
      "日期",
      "星期",
      "是否校历寒暑假",
      "校历假期类型",
      "是否法定节假日假期",
      "法定节假日名称",
      "是否调休工作日",
    ],
  },
  {
    source: "a_hupan_sales.csv",
    output: "a_hupan_sales_20260611_20260824.csv",
    sheetName: "湖畔销售",
    expectedRows: 22800,
    headers: [
      "年",
      "月",
      "日",
      "餐次",
      "星期",
      "是否校历寒暑假",
      "校历假期类型",
      "是否法定节假日假期",
      "法定节假日名称",
      "是否调休工作日",
      "气温（℃）",
      "相对湿度（%）",
      "降水量（毫米）",
      "10米风速（公里/小时）",
      "菜品名称",
      "菜品总销量",
      "菜品售卖单位",
      "订单数量",
      "菜品单价",
      "销售额",
      "平均取菜量",
      "食堂名称",
    ],
  },
];

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = value instanceof Date
    ? value.toISOString().slice(0, 10)
    : String(value);
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function toCsv(matrix) {
  return `\uFEFF${matrix.map((row) => row.map(csvCell).join(",")).join("\r\n")}\r\n`;
}

await fs.mkdir(outputRoot, { recursive: true });
await fs.mkdir(previewRoot, { recursive: true });
const verification = [];

for (const item of files) {
  const sourceText = await fs.readFile(path.join(sourceRoot, item.source), "utf8");
  const workbook = await Workbook.fromCSV(sourceText, { sheetName: item.sheetName });
  const sheet = workbook.worksheets.getItem(item.sheetName);
  const matrix = sheet.getUsedRange(true).values;
  if (typeof matrix[0]?.[0] === "string") {
    matrix[0][0] = matrix[0][0].replace(/^\uFEFF/, "");
  }
  const headers = matrix[0].map((value) => String(value));
  if (JSON.stringify(headers) !== JSON.stringify(item.headers)) {
    throw new Error(`${item.output} 字段不符合更新模块要求`);
  }
  if (matrix.length !== item.expectedRows) {
    throw new Error(`${item.output} 行数异常：${matrix.length}`);
  }
  await fs.writeFile(path.join(outputRoot, item.output), toCsv(matrix), "utf8");

  const inspected = await workbook.inspect({
    kind: "table",
    sheetId: item.sheetName,
    range: `A1:${item.sheetName === "湖畔销售" ? "V" : item.sheetName === "校历" ? "G" : "E"}4`,
    include: "values,formulas",
    tableMaxRows: 4,
    tableMaxCols: item.headers.length,
    maxChars: 5000,
  });
  const preview = await workbook.render({
    sheetName: item.sheetName,
    range: `A1:${item.sheetName === "湖畔销售" ? "V" : item.sheetName === "校历" ? "G" : "E"}12`,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(previewRoot, `${item.sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
  verification.push({
    output: item.output,
    rows: matrix.length - 1,
    columns: matrix[0].length,
    inspection: inspected.ndjson,
  });
}

await fs.writeFile(
  path.join(previewRoot, "verification.json"),
  JSON.stringify(verification, null, 2),
  "utf8",
);

process.stdout.write(`${outputRoot}\n`);
