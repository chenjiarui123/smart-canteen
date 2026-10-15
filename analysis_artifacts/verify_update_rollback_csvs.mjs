import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const projectRoot = path.resolve(".");
const dataRoot = path.join(projectRoot, "smart_canteen_web", "data");
const previewRoot = "/private/tmp/smart_canteen_update_rollback_previews";
const mode = process.argv[2] ?? "active";

const sources = mode === "backup"
  ? [
      {
        name: "湖畔运行期销售",
        file: path.join(dataRoot, "backups", "a_hupan", "runtime_actual_sales_20260824_205808_766201.csv"),
        expectedRows: 1,
        expectedColumns: 22,
      },
      {
        name: "天气",
        file: path.join(dataRoot, "backups", "features", "weather_20260824_205747_717005.csv"),
        expectedRows: 512,
        expectedColumns: 5,
      },
      {
        name: "校历",
        file: path.join(dataRoot, "backups", "features", "calendar_20260824_205731_068313.csv"),
        expectedRows: 512,
        expectedColumns: 7,
      },
    ]
  : [
      {
        name: "湖畔运行期销售",
        file: path.join(dataRoot, "datasets", "a_hupan", "runtime_actual_sales.csv"),
        expectedRows: 1,
        expectedColumns: 22,
      },
      {
        name: "天气",
        file: path.join(dataRoot, "features", "weather.csv"),
        expectedRows: 512,
        expectedColumns: 5,
      },
      {
        name: "校历",
        file: path.join(dataRoot, "features", "calendar.csv"),
        expectedRows: 512,
        expectedColumns: 7,
      },
    ];

await fs.mkdir(previewRoot, { recursive: true });
const result = [];
for (const item of sources) {
  const text = await fs.readFile(item.file, "utf8");
  const workbook = await Workbook.fromCSV(text, { sheetName: item.name });
  const sheet = workbook.worksheets.getItem(item.name);
  const used = sheet.getUsedRange(true);
  const matrix = used.values;
  if (matrix.length !== item.expectedRows || matrix[0].length !== item.expectedColumns) {
    throw new Error(`${item.name} 行列数不符合回滚目标：${matrix.length}×${matrix[0].length}`);
  }
  const inspected = await workbook.inspect({
    kind: "table",
    sheetId: item.name,
    range: item.name === "湖畔运行期销售" ? "A1:V4" : item.name === "天气" ? "A1:E4" : "A1:G4",
    include: "values,formulas",
    tableMaxRows: 4,
    tableMaxCols: item.expectedColumns,
    maxChars: 6000,
  });
  const preview = await workbook.render({
    sheetName: item.name,
    range: item.name === "湖畔运行期销售" ? "A1:V4" : item.name === "天气" ? "A1:E12" : "A1:G12",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(previewRoot, `${mode}_${item.name}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
  result.push({
    name: item.name,
    file: item.file,
    rowsIncludingHeader: matrix.length,
    columns: matrix[0].length,
    inspection: inspected.ndjson,
  });
}

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
