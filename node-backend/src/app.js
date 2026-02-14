import express from "express"
import cors from "cors"
import cookieParser from "cookie-parser"
import { errorHandler } from "./middlewares/errorHandler.middleware.js"

const app = express()

app.use(cors({
    origin: process.env.CORS_ORIGIN,
    credentials: true,
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
}));

app.use(express.json({ limit: "16kb" }))
app.use(express.urlencoded({ extended: true, limit: "16kb" }))
app.use(express.static("public"))
app.use(cookieParser())


//routes import
import userRouter from './routes/user.routes.js'
import notificationRouter from './routes/notification.route.js'
import projectRouter from "./routes/project.route.js"
import modelRouter from "./routes/model.route.js"
import datasetRouter from "./routes/dataset.route.js"

//routes declaration
app.use("/api/v1/users", userRouter)
app.use("/api/v1/notifications", notificationRouter)
app.use("/api/v1/projects", projectRouter)
app.use("/api/v1/models", modelRouter)
app.use("/api/v1/datasets", datasetRouter)
app.use("/api/v1/setting", datasetRouter)
app.use('/public', express.static('public'))

app.use(errorHandler)

export { app }