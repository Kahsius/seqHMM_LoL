# ====================================================-- #
# ============= Debuts de systeme d'analyse ============ #
# ====================================================-- #
# ======================== TODO ======================== #
# ====================================================-- #

# Analyse état part état
# Pour chaque état, on regarde les probabilités d'émission associées
# Pour avoir une idée de ce que l'état représente
# Pour l'instant on ne regarde que les etats binaires

fmegahmm <- megahmms$fitted.hmms[[1]]
states <- vector("list", N_STATES2)
for(i in 1:N_STATES2){
  if(MODE_PER_TEAM){
    state <- matrix(0, N_STATES, N_CLUSTERS_PROBAS)
    for(j in 1:N_STATES){
      state[j,] <- fmegahmm$model$emission_probs[[j]][i,]
    }       
  }else{
    state <- matrix(0, N_TEAM*N_STATES, N_CLUSTERS_PROBAS)
    for(j in 1:(N_TEAM*N_STATES)){
      state[j,] <- fmegahmm$model$emission_probs[[j]][i,]
    }    
  }
  rownames(state) <- names(fmegahmm$model$emission_probs)
  states[[i]] <- state
}
coeffs <- matrix(nrow = N_STATES2, ncol = N_STATES*N_TEAM)
segments_probas_mean <- vector("numeric", N_CLUSTERS_PROBAS)
for(i in 1:N_CLUSTERS_PROBAS){
  if(i!=1){
    segments_probas_mean[i] <- sum(segments_probas[(i-1):i])/2
  } else {
    segments_probas_mean[i] <- 0
  }
}
for(i in 1:N_STATES2){
  coef <- vector("numeric", N_STATES*N_TEAM)
  for(j in 1:N_TEAM){
    coef.team <- vector("numeric", N_STATES)
    for(k in 1:N_STATES){
      coef.team[k] <- sum(states[[i]][k+(j-1)*N_STATES,]*segments_probas_mean)
    }
    coef.team <- coef.team/sum(coef.team)
    coef[1:N_STATES+(j-1)*N_STATES] <- coef.team
  }
  coeffs[i,] <- coef
}

# 
states.t1 <- vector("list", N_STATES)
for(i in 1:N_STATES){
  state <- matrix(0,length(vocab),2)
  for(j in 1:length(vocab)){
    state[j,] <- hmm.per.team$fitted.hmms[[1]]$model$emission_probs[[j]][i,]
  }
  rownames(state) <- names(hmm.per.team$fitted.hmms[[1]]$model$emission_probs)
  states.t1[[i]] <- state
}
# 
states.t2 <- vector("list", N_STATES)
for(i in 1:N_STATES){
  state <- matrix(0,length(vocab),2)
  for(j in 1:length(vocab)){
    state[j,] <- hmm.per.team$fitted.hmms[[2]]$model$emission_probs[[j]][i,]
  }
  rownames(state) <- names(hmm.per.team$fitted.hmms[[2]]$model$emission_probs)
  states.t2[[i]] <- state
}
#
images <- vector("list", N_STATES2)
for(i in 1:N_STATES2){
  image <- matrix(0, 2*length(vocab),2)
  for(j in 1:length(vocab)){
    s <- c(0,0)
    for(k in 1:N_STATES){
      s <- s + coeffs[i,k]*states.t1[[k]][j,]
    }
    image[j,] <- s
  }
  for(j in 1:length(vocab)){
    s <- c(0,0)
    for(k in 1:N_STATES){
      s <- s + coeffs[i,k+N_STATES]*states.t2[[k]][j,]
    }
    image[j+length(vocab),] <- s
  }
  n1 <- as.vector(sapply(names(hmm.per.team$fitted.hmms[[1]]$model$emission_probs), function(e){
    return(paste("Player 1",substring(e,9)))
  }))
  n2 <- as.vector(sapply(names(hmm.per.team$fitted.hmms[[2]]$model$emission_probs), function(e){
    return(paste("Player 2",substring(e,9)))
  }))
  rownames(image) <- c(n1,n2)
  images[[i]] <- image
}

# comparatif strats
comp <- vector("list", N_STATES2)
l <- length(vocab)
for(i in 1:N_STATES2){
  comp[[i]] <- images[[i]][1:l,]-images[[i]][1:l+l,]
}
