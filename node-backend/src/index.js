// require('dotenv').config({path: './env'})
import dotenv from "dotenv"
import connectDB from "./db/index.js";
import { app } from './app.js'
import logger from "./utils/logger.js"

dotenv.config({
    path: './.env'
})


connectDB()
    .then(() => {
        logger.info("Database connected successfully")
        app.listen(process.env.PORT || 8000, () => {
            logger.info(`Server started on port ${process.env.PORT || 8000}`)
        })
    })
    .catch((err) => {
        logger.error("Database connection failed", { error: err.message })
    })