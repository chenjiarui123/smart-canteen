import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const buildRoot = "/private/tmp/smart_canteen_update_20260824_build";
const outputDir = path.resolve("outputs/data_update_test_20260824");
const outputPath = path.join(
  outputDir,
  "smart_canteen_update_test_20260611_20260824.xlsx",
);

const workbook = Workbook.create();
const notes = workbook.worksheets.add("使用说明");
notes.showGridLines = false;
notes.getRange("A1:F1").merge();
notes.getRange("A1").values = [["智慧食堂数据更新模块测试数据"]];
notes.getRange("A1:F1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
notes.getRange("A1:F1").format.rowHeight = 30;
notes.getRange("A3:B12").values = [
  ["数据性质", "模拟测试数据，不代表真实经营结果，请勿写入正式环境"],
  ["覆盖日期", "2026-06-11 至 2026-08-24，共75天"],
  ["上传顺序", "1. 天气；2. 校历；3. A湖畔销售；4. F基础园销售"],
  ["天气工作表", "天气"],
  ["校历工作表", "校历"],
  ["A／湖畔销售工作表", "A湖畔销售"],
  ["F／基础园销售工作表", "F基础园销售"],
  ["天气生成", "采用2025年同期天气；缺失日按相邻日期插值"],
  ["校历假设", "2026-07-21起标记暑假；2026-06-19至06-21标记端午节测试场景"],
  ["销售生成", "按近期同星期历史菜单轮换，并施加确定性的日期、周末与暑假缩放"],
];
notes.getRange("A3:A12").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#1F1F1F" },
};
notes.getRange("B3:B12").format = { wrapText: true };
notes.getRange("A3:B12").format.borders = {
  preset: "all",
  style: "thin",
  color: "#B8C7D1",
};
notes.getRange("A14:F16").merge();
notes.getRange("A14").values = [[
  "上传该多工作表文件时，系统会要求选择工作表。销售更新前必须先提交天气和校历，否则销售预览应被阻止。这一行为属于预期测试结果。",
]];
notes.getRange("A14:F16").format = {
  fill: "#FFF2CC",
  font: { color: "#7F6000" },
  wrapText: true,
  verticalAlignment: "center",
};
notes.getRange("A3:A16").format.columnWidth = 24;
notes.getRange("B3:B16").format.columnWidth = 72;
notes.freezePanes.freezeRows(1);

const imports = [
  ["weather.csv", "天气"],
  ["calendar.csv", "校历"],
  ["a_hupan_sales.csv", "A湖畔销售"],
  ["f_jichuyuan_sales.csv", "F基础园销售"],
];

for (const [filename, sheetName] of imports) {
  const csvText = await fs.readFile(path.join(buildRoot, filename), "utf8");
  const parsedWorkbook = await Workbook.fromCSV(csvText, { sheetName });
  const parsedSheet = parsedWorkbook.worksheets.getItem(sheetName);
  const parsedRange = parsedSheet.getUsedRange(true);
  const matrix = parsedRange.values;
  if (typeof matrix[0]?.[0] === "string") {
    matrix[0][0] = matrix[0][0].replace(/^\uFEFF/, "");
  }
  const sheet = workbook.worksheets.add(sheetName);
  sheet
    .getRangeByIndexes(0, 0, matrix.length, matrix[0].length)
    .values = matrix;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange(true);
  used.getRow(0).format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  used.getRow(0).format.rowHeight = 30;
  if (sheetName === "天气") {
    for (let index = 0; index < 5; index += 1) {
      used.getColumn(index).format.columnWidth = 21;
    }
  } else if (sheetName === "校历") {
    for (let index = 0; index < 7; index += 1) {
      used.getColumn(index).format.columnWidth = 20;
    }
  } else {
    for (let index = 0; index < 22; index += 1) {
      used.getColumn(index).format.columnWidth = 16;
    }
    used.getColumn(14).format.columnWidth = 28;
    used.getColumn(18).format.columnWidth = 19;
    used.getColumn(21).format.columnWidth = 24;
  }
}

await fs.mkdir(outputDir, { recursive: true });
for (const sheetName of ["使用说明", "天气", "校历", "A湖畔销售", "F基础园销售"]) {
  const range = sheetName === "使用说明" ? "A1:F16" : "A1:V12";
  const preview = await workbook.render({
    sheetName,
    range,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(buildRoot, `${sheetName}_preview.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const inspected = await workbook.inspect({
  kind: "sheet,table",
  include: "id,name",
  maxChars: 3000,
  tableMaxRows: 3,
  tableMaxCols: 6,
});
await fs.writeFile(
  path.join(buildRoot, "workbook_inspect.ndjson"),
  inspected.ndjson,
  "utf8",
);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(buildRoot, "formula_error_scan.ndjson"),
  errors.ndjson,
  "utf8",
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
process.stdout.write(`${outputPath}\n`);
