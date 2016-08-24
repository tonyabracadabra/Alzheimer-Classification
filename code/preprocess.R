setwd("/Volumes/TONY/LabWW/Alzheimer/GSE44772")

library(GEOquery)
require(xgboost)

gse44772 = getGEO(filename = "GSE44772_family.soft")

gsms = gse44772@gsms

annotation = gse44772@gpls$GPL4372@dataTable@table

# get ordered probe ID
ID_REF = annotation$ID

# dataMatrix = within(gsms$GSM751160@dataTable@table, rm(VALUE))
ID_REF = intersect(gsms$GSM1090267@dataTable@table$ID_REF, ID_REF)

for (gsm in gsms) {
    stopifnot(setequal(intersection, gsm@dataTable@table$ID_REF))
}

X = data.frame(ID_REF)
Y = c()
count = 1
for (gsm in gsms) {
    if ((count %% 10) == 1) {
        print(paste0("Has Processed ", count))
    }
    
    table = gsm@dataTable@table

    region = strsplit(gsm@header$title, "_")[[1]][2]
    disease = strsplit(gsm@header$characteristics_ch2[1], " ")[[1]][2]

    # print(title)
    Y = c(Y, paste(disease, region, sep="_"))
    # print(y)
    # Merge data matrix
    X = merge(X, table, by = "ID_REF")
    count = count + 1
}

X$ID_REF = NULL
X = t(data.matrix(X))
Y = data.matrix(Y)

mapY = c(1:6)
names(mapY) = unique(Y)

# Convert to integers
numY = data.matrix(apply(Y, 1, function(x) mapY[x])-1)

processedData = list(X, numY)
names(processedData) = c("X", "Y")

sub = sample(nrow(X), floor(nrow(X) * 0.8))

x.train = processedData$X[sub, ]
y.train = processedData$Y[sub, ]

x.test = processedData$X[-sub, ]
y.test = processedData$Y[-sub, ]

dataXGB.train = xgb.DMatrix(data = x.train, label = y.train, missing=NaN)
dataXGB.test = xgb.DMatrix(data = x.test, label = y.test, missing=NaN)

model.xgb = xgb.train(data = dataXGB.train, nrounds = 100, objective = "multi:softmax", num_class = 6)

y.predict = predict(model, newdata = dataXGB.test)

# Confusion matrix
cm = confusion(y.predict, y.test)

# Build random forest model
dataRF.train = split(x.train, rep(1:ncol(x.train), each = nrow(x.train)))
dataRF.train$y = y.train
model.rf = randomForest(y ~ ., data = dataRF.train, ntree = 100)
predict(model.rf, )