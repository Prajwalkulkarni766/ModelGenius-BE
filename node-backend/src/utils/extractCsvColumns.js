import fs from "fs";
import csv from "csv-parser";

export const extractCsvColumns = (filePath) => {
  return new Promise((resolve, reject) => {
    const stream = fs.createReadStream(filePath);
    stream
      .pipe(csv())
      .on("headers", (headers) => {
        stream.destroy();
        resolve(headers);
      })
      .on("error", (error) => {
        reject(error);
      });
  });
};
