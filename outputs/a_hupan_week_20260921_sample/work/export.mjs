import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook } from '@oai/artifact-tool';

const work = path.dirname(fileURLToPath(import.meta.url));
const {rows: inputRows} = JSON.parse(await fs.readFile(path.join(work, 'sample.json'), 'utf8'));
const rows = inputRows.map(row => row.slice(1));
const wb = Workbook.create();
const sheet = wb.worksheets.add('周一午餐');
sheet.getRange(`A1:G${rows.length}`).values = rows;
sheet.getRange(`A1:G${rows.length}`).format.font = {name:'Arial', size:11};
sheet.getRange(`A1:G${rows.length}`).format.rowHeight = 24;
for (const [col, width] of [['A',18],['B',14],['C',14],['D',30],['E',23],['F',20],['G',14]]) {
  sheet.getRange(`${col}1`).format.columnWidth = width;
}
sheet.getRange(`C2:C${rows.length}`).setNumberFormat('@');
sheet.getRange(`E2:F${rows.length}`).setNumberFormat('0.00');
wb.recalculate();
console.log((await wb.inspect({kind:'table', range:'周一午餐!A1:G5', include:'values', tableMaxRows:5, tableMaxCols:7})).ndjson);
const values = sheet.getRange(`A1:G${rows.length}`).values;
if (JSON.stringify(values) !== JSON.stringify(rows)) throw Error('Output values changed');
const quote = value => '"' + String(value ?? '').replaceAll('"','""') + '"';
await fs.writeFile(path.join(work,'../A湖畔_周一_午餐.csv'), '\uFEFF'+values.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n');
const preview = await wb.render({sheetName:'周一午餐',range:'A1:G15',scale:1,format:'png'});
await fs.writeFile(path.join(work,'preview.png'),new Uint8Array(await preview.arrayBuffer()));
console.log(`Exported ${rows.length} rows`);
