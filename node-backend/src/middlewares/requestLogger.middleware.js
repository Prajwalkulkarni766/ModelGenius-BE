import logger, { maskSensitiveFields } from "../utils/logger.js";

const requestLogger = (req, res, next) => {
  const start = Date.now();
  const { method, originalUrl, ip } = req;
  const userAgent = req.get("user-agent") || "";
  const clientIp = req.ip || req.connection?.remoteAddress || ip;

  res.on("finish", () => {
    const { statusCode } = res;
    const duration = Date.now() - start;

    const logData = {
      method,
      url: originalUrl,
      status: statusCode,
      duration: `${duration}ms`,
      ip: clientIp,
      userAgent,
    };

    if (req.body && Object.keys(req.body).length > 0) {
      logData.payload = maskSensitiveFields(req.body);
    }

    if (statusCode >= 500) {
      logger.error("Request failed", logData);
    } else if (statusCode >= 400) {
      logger.warn("Request warning", logData);
    } else {
      logger.info("Request completed", logData);
    }
  });

  next();
};

export default requestLogger;