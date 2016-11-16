
# ====================================================-- #
# ============= Debuts de systeme d'analyse ============ #
# ====================================================-- #
# ======================== TODO ======================== #
# ====================================================-- #

# Uniquement si les canaux sont binaires, à améliorer pour le faire pour tout alphabet
# states.T <- vector("list", N_TEAM)
# for(k in 1:N_TEAM){
#   states.T[[k]] <- vector("list", N_STATES)
#   for(i in 1:N_STATES){
#     state <- matrix(0,length(vocab),2)
#     for(j in 1:length(vocab)){
#       state[j,] <- fmhmm$model$emission_probs[[k]][[j]][i,]
#     }
#     rownames(state) <- names(h$model$emission_probs)
#     states.T[[k]][[i]] <- state
#     colnames(states.T[[k]][[i]]) <- c("FALSE","TRUE")
#     rownames(states.T[[k]][[i]]) <- vocab
#   }  
# }

if(MODE_PER_TEAM){
  ihmm <- 1
  
  # Pour le mega hmm
  fmegahmm <- megahmms$fitted.hmms[[ihmm]]
  states <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    state <- matrix(0,N_STATES,N_CLUSTERS_PROBAS)
    for(j in 1:N_STATES){
      state[j,] <- fmegahmm$model$emission_probs[[j]][i,]
    }
    rownames(state) <- names(fmegahmm$model$emission_probs)
    states[[i]] <- state
    colnames(states[[i]]) <- 1:N_CLUSTERS_PROBAS
  }
  
  hmm <- hmm.per.team$fitted.hmms[[ihmm]]
  images <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    image <- matrix(0, nrow = 15, ncol = 15)
    for(j in 1:N_STATES){
      coef <- sum(states[[i]][j,]*segments_probas)
      xx <- hmm$model$emission_probs$x[j,]
      yy <- hmm$model$emission_probs$y[j,]
      mx <- replicate(15,xx)
      my <- t(replicate(15,yy))
      image <- image + coef*mx*my
    }
    images[[i]] <- image
  }
} else {
  # Pour le mega hmm unique
  model <- fmegahmm$model
  states <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    state <- matrix(0,2*N_STATES,N_CLUSTERS_PROBAS)
    for(j in 1:(2*N_STATES)){
      state[j,] <- model$emission_probs[[j]][i,]
    }
    rownames(state) <- names(model$emission_probs)
    states[[i]] <- state
    colnames(states[[i]]) <- 1:N_CLUSTERS_PROBAS
  }
  
  hmm1 <- hmm.per.team$fitted.hmms[[1]]
  images1 <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    image <- matrix(0, nrow = 15, ncol = 15)
    for(j in 1:N_STATES){
      coef <- sum(states[[i]][j,]*segments_probas)
      xx <- hmm1$model$emission_probs$x[j,]
      yy <- hmm1$model$emission_probs$y[j,]
      mx <- replicate(15,xx)
      my <- t(replicate(15,yy))
      image <- image + coef*mx*my
    }
    images1[[i]] <- image
  }
  
  hmm2 <- hmm.per.team$fitted.hmms[[2]]
  images2 <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    image <- matrix(0, nrow = 15, ncol = 15)
    for(j in 1:N_STATES){
      coef <- sum(states[[i]][j+N_STATES,]*segments_probas)
      xx <- hmm2$model$emission_probs$x[j,]
      yy <- hmm2$model$emission_probs$y[j,]
      mx <- replicate(15,xx)
      my <- t(replicate(15,yy))
      image <- image + coef*mx*my
    }
    images2[[i]] <- image
  }
  
  images <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    images[[i]] <- images1[[i]] + images2[[i]]
  }
  
  images_inf1 <- vector("list", N_STATES)
  hmm1 <- hmm.per.team$fitted.hmms[[1]]
  for(i in 1:N_STATES){
    xx <- hmm1$model$emission_probs$x[i,]
    yy <- hmm1$model$emission_probs$y[i,]
    mx <- replicate(15,xx)
    my <- t(replicate(15,yy))
    images_inf1[[i]] <-mx*my
  }

  images_inf2 <- vector("list", N_STATES)
  hmm2 <- hmm.per.team$fitted.hmms[[2]]
  for(i in 1:N_STATES){
    xx <- hmm2$model$emission_probs$x[i,]
    yy <- hmm2$model$emission_probs$y[i,]
    mx <- replicate(15,xx)
    my <- t(replicate(15,yy))
    images_inf2[[i]] <-mx*my
  }
}