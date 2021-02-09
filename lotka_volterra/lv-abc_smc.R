# BEFORE RUNNING THIS FILE REMEMBER TO SET THE CURRENT WORKING DIRECTORY TO THE SAME LOCATION OF THIS FILE.
# HOWEVER THIS IS IMPORTANT ONLY IF THE OPTION VERBOSE=TRUE IS ACTIVATED, AS INTERMEDIATE FILES ARE GOING TO BE CREATED IN THE 
# WORKING DIRECTORY

# this is a case study for SMC-ABC applied to the Lotka-Volterra model as studied in:
# https://arxiv.org/pdf/1805.07226.pdf
# https://arxiv.org/pdf/1605.06376.pdf

setwd("~/Box Sync/LV_EasyABC/LV_Samuel_numpart10000")

# WARNING: for us the first coordinate of the LV state is PREY, the second one is PREDATOR.
# this important to be taken into account as other sources may invert the order

require(smfsb)     # for the model simulator
require(parallel)  # actually unused
require(chemometrics) # only sueful to conmputed trimmed standard deviations (to standardize the ABCsummaries)
require(EasyABC)   # ANC inference engine
options(mc.cores=4)

N= 1000  # number of particles. 
bs=1e3   # number of samples for the pilot run (prior predictive samples)
message(paste("N =",N," | bs =",bs))


# we define some summary stats for a univariate time series - the mean, the (log) variance, and the first two auto-correlations.
ssinit <- function(vec)
{
  ac23=as.vector(acf(vec,lag.max=2,plot=FALSE)$acf)[2:3]
  c(mean(vec),log(var(vec)+1),ac23)
}

# Once we have this, we can define some stats for a bivariate time series by combining the stats for the two component series, along with the cross-correlation between them.
ssi <- function(ts)
{
  c(ssinit(ts[,1]),ssinit(ts[,2]),cor(ts[,1],ts[,2]))
}


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Define the Lotka Volterra model.
# Th following vignette is useful: https://cran.r-project.org/web/packages/smfsb/vignettes/smfsb.pdf
# HOWEVER, the example above is for a 3-reactions LV model.
# What we want is a 4-reactions LV as in the papers below:
# references are https://arxiv.org/pdf/1605.06376.pdf
# and also https://arxiv.org/pdf/1805.07226.pdf
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# x1 is prey, x2 is predator
Pre  <- matrix(data=c(0,0,1,1,1,1,0,1),nrow=4)
Post  <- matrix(data=c(0,0,2,0,2,0,0,2),nrow=4)
init <- c(50,100)  # initial populations size for PREY and PREDATORS respectively
h<-function (x, t, th) # true values are th = c(th1 = 0.01, th2 = 0.5, th3 = 1, th4 = 0.01)
{
  with(as.list(c(x, th)), {
    return(c(th1 * x1 * x2, th2 * x2, th3 * x1, th4 * x1 * x2))
  })
}
LV <- list("Pre"=Pre,"Post"=Post,"M"=init, "h"=h)
## create a stepping function
stepLV = StepGillespie(LV)
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#::::: PILOT RUN FROM THE PRIOR PREDICTIVE. ONLY USEFUL TO COMPUTE STANDARD DEVIATIONS AND MEANS OF SUMMARIES
# the part below is a bit computer intensive
# if you prefer to skip it, the relevant means and SD for the summaries are pasted below:
# means_trim = c(0.36797185,  2.88176212,  0.04942526,  0.00949382, 91.42669702,  7.48977408,  0.88494884,  0.78470815,  0.07959541)
# sds_trim = c(0.12564098,   0.10529177,   0.14566593 ,  0.04150598 ,126.36148388   ,2.21371448 ,  0.13243152   ,0.23219528 ,  0.32433566)
prior=cbind(th1=exp(runif(bs,-5,2)),th2=exp(runif(bs,-5,2)),th3=exp(runif(bs,-5,2)),th4=exp(runif(bs,-5,2)))
rows=lapply(1:bs,function(i){prior[i,]})
samples=mclapply(rows,function(th){simTs(c(x1=50,x2=100),0,30.2,0.2,stepLV,th)})
sumstats=mclapply(samples,ssi)
# identify problematic cases giving rise to "Nan"
idnan <- NULL
for(ii in 1:bs){
  sumnan <- sum(is.nan(sumstats[[ii]]))  # the number of NaN's in the current summary
  if(sumnan>0){
    idnan <- c(idnan,ii)  # collect here indeces of simulations containing NaNs
  }
}
sds_trim=apply(sapply(sumstats[-idnan],c),1,sd_trim,trim=0.10)  # compute trimmed SD for not-NaNs cases
means_trim=apply(sapply(sumstats,c),1,mean,na.rm=T,trim=0.10) # trim on cases without NaN
print(sds_trim)
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# load data
lv_data <- read.table("~/Box Sync/LV_EasyABC/LV_Samuel/lv_data", quote="\"", comment.char="")
# oberved summaries. Also, we standardise them
sso=(ssi(lv_data)-means_trim)/sds_trim

# Model simulator for the summaries (also standardised)
LV_model<-function(logth){
  th = exp(logth)
  trajectory <- simTs(c(x1=50,x2=100),0,30.2,0.2,stepLV,c(th1=th[1],th2=th[2],th3=th[3],th4=th[4]))
  (ssi(trajectory)-means_trim) / sds_trim
}


# priors
LV_prior=list(c("unif",-5,2),c("unif",-5,2),c("unif",-5,2),c("unif",-5,2))

# Delmoral et al ABC-SMC 
tolerance <- 0.00001
set.seed(123)
ABC_smc<-ABC_sequential(method="Delmoral",model=LV_model,prior=LV_prior, nb_simul=N, nb_threshold = 0.5*N, summary_stat_target=sso, alpha=0.7, M=1, tolerance_target = tolerance, verbose =TRUE)

# NOTICE, MANY FILES ARE CREATED IN THE WORKING DIRECTORY. FILES NAMED OUTPUT_STEPX CONTAIN THE POSTERIOR DRAWS IN THE FOLLOWING ORDER:
# DRAWS FROM LOG-THETA1 ARE IN COLUMN 2, DRAWS FROM LOG-THETA2 IN COLUMN3 ETC. Afterwards follow simulated summaries 



# Beaumont's et al method
#set.seed(123)
#ABC_smc<-ABC_sequential(method="Beaumont",model=LV_model,prior=LV_prior, nb_simul=N, summary_stat_target=sso, tolerance = c(seq(50,1)), verbose =TRUE)



#set.seed(123)
#ABC_smc<-ABC_sequential(method="Drovandi",model=LV_model,prior=LV_prior, nb_simul=N, summary_stat_target=sso, alpha=0.3, first_tolerance_level_auto = F, tolerance_tab =c(110,95), verbose =TRUE)

