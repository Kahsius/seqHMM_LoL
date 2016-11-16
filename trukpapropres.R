# TRUKPAPROPRES #
#Vérification que les états sont bien représentés
represent <- matrix(NA, nrow=N_TEAM*N_STATES, ncol = N_CLUSTERS_PROBAS)
for(i in 1:(N_TEAM*N_STATES)){
  represent[i,] <- colSums(fmegahmm$model$emission_probs[[i]])
}
# ================================================================================================

n <- 60
s <- sqrt(n)
for(i in floor(s):1){
  if(n%%i==0){
    nr <- i
    nc <- n/i
    break
  }
}

# Mega HMM
layout(t(matrix(1:n, nrow = nc, ncol = nr)))
for(i in 1:n){myImagePlot(images[[i]])}
layout(1)

# Comparatifs de stratégie
seuil = 0.2
layout(t(matrix(1:40, nrow = 8, ncol = 5)))
for(i in 1:40){
  tmp <- comp[[i]]
  tmp[abs(tmp)<=seuil] <- 0
  myImagePlot(tmp)
}
layout(1)

# HMM1
layout(t(matrix(1:20, nrow = 4, ncol = 5)))
for(i in 1:20){myImagePlot(images_inf1[[i]])}
layout(1)

# HMM2
layout(t(matrix(1:20, nrow = 4, ncol = 5)))
for(i in 1:20){myImagePlot(images_inf2[[i]])}
layout(1)
# ================================================================================================

v <- hidden_paths(hmm1$model)
v <- as.matrix(v)
t <- table(v)
r <- t/sum(t)

b <- c("State 1", "State 3", "State 5", "State 9", "State 12", "State 15", "State 19")
rr <- vector("numeric", n_matches)
for(i in 1:n_matches){
  rr[i] <- length(which(v[1:5+5*(i-1),] %in% b))
}

# ================================================================================================

i <- 2
model <- megahmms.test

#Jouli animation
viterbi <- as.matrix(hidden_paths(model))
v1 <- viterbi[i,]
for(i in 1:length(v1)){
  index <- as.numeric(strsplit(v1[i], split=" ")[[1]][2])
  myImagePlot(images[[index]])
  Sys.sleep(.3)
}

# Enregistrement dans un fichier
saveGIF({
  i <- 3
  viterbi <- as.matrix(hidden_paths(fmegahmm$model))
  v <- viterbi[i,]
  for(i in 1:length(v)){
    index <- as.numeric(strsplit(v[i], split=" ")[[1]][2])
    myImagePlot(images[[index]])
  }
})

# ================================================================================================
# Essais test/apprentissage
# Séparation des deux équipes
seqdef.test <- vector("list")
seqdef.test$teams <- vector("list", N_TEAM)
d <- dim(matrice.global.test[[1]])
n_matches_test <- d[1]/N_P_PER_TEAM/N_TEAM
for(i in 1:N_TEAM){
  seqdef.test$teams[[i]]$matrices <- vector("list", length(matrice.global.test))
  seqdef.test$teams[[i]]$seq <- vector("list", length(matrice.global.test))
  for(j in 1:length(matrice.global.test)){
    seqdef.test$teams[[i]]$matrices[[j]] <- matrix(NA, nrow = d[1]/2, ncol = d[2])
    for(k in 1:(d[1]/10)){
      seqdef.test$teams[[i]]$matrices[[j]][1:N_P_PER_TEAM+(k-1)*N_P_PER_TEAM,] <- matrice.global.test[[j]][1:N_P_PER_TEAM+(k-1)*N_P_PER_TEAM*N_TEAM+(i-1)*N_P_PER_TEAM,]
    }
  }
}
for(i in 1:length(matrice.global.test)){
  for(j in 1:N_TEAM){
    seqdef.test$teams[[j]]$seq[[i]] <- seqdef(
      data = seqdef.test$teams[[j]]$matrice[[i]],
      alphabet = alphabets[[i]])      
  }
}
hmms.test <- vector("list",N_TEAM)
for(i in 1:N_TEAM){
  # Uniquement pour les positions
  hmms.test[[i]] <- transfer_data(seqdef.test$teams[[i]]$seq, hmm.per.team$fitted.hmms[[i]]$model)
}

# RECOPIE MISE EN FORME DES DONNEES FORWARD
print("--- Recuperation des forwards probabilities")
r <- vector("list", N_TEAM)
for(i in 1:N_TEAM){
  r[[i]] <- forward_backward(hmms.test[[i]], forward_only = T, threads = N_THREADS)
}
# m_forward[[équipe]][[séquence d'un joueur de cette team]]
m_forward <- vector("list", N_TEAM)
for(i in 1:N_TEAM){
  m_forward[[i]] <- vector("list", n_matches_test*N_P_PER_TEAM)
}
for(i in 1:n_matches_test){
  for(j in 1:N_P_PER_TEAM){
    for(k in 1:N_TEAM){
      m_forward[[k]][[(i-1)*N_P_PER_TEAM+j]] <- r[[k]]$forward_probs[,,(i-1)*N_P_PER_TEAM+j]
    }
  }
}
print("--- Mise en forme et sommage")
m_all_forward <- vector("list", n_matches_test)
for(i in 1:n_matches_test){
  t <- matrix(data = 0, nrow = N_TEAM*N_STATES, ncol = TMAX)
  for(l in 1:TMAX){
    for(j in 1:N_TEAM){
      v <- vector("numeric", N_STATES)
      for(k in 1:N_P_PER_TEAM){
        v <- v + m_forward[[j]][[k+(i-1)*N_P_PER_TEAM]][,l]
      }
      t[1:N_STATES+(j-1)*N_STATES,l] <- v
    }    
  }
  rn <- as.vector(matrix(NA, nrow = 1, ncol = N_TEAM*N_STATES))
  for(k in 1:N_TEAM){
    for(j in 1:N_STATES){
      rn[j+(k-1)*N_STATES] <- paste("Team",k,"State",j)
    }
  }
  cn <- as.vector(matrix(NA, nrow = 1, ncol = TMAX))
  for(k in 1:TMAX){
    cn[k] <- paste("T",k,sep="")
  }
  colnames(t) <- cn
  rownames(t) <- rn
  m_all_forward[[i]] <- t/N_P_PER_TEAM
}

print("--- Reorganisation des donnees suivant les canaux")
mmm <- vector("list", N_TEAM*N_STATES)
mmm2 <- vector("list", N_TEAM*N_STATES)
for (i in 1:(N_TEAM*N_STATES)){
  mtmp <- matrix(0, n_matches_test, TMAX)
  rownames(mtmp) <- 1:n_matches_test
  colnames(mtmp) <- colnames(r$forward_probs[,,1])
  for (j in 1:n_matches_test){
    mtmp[j,] <- m_all_forward[[j]][i,]
  }
  mmm[[i]] <- mtmp
}

# Reformatage de mmm pour avoir les segments au lieu des probas
print("--- Reformatage des probabilites a l'aide des segments")
for(i in 1:(N_TEAM*N_STATES)){
  mtmp <- mmm[[i]]
  for(j in N_CLUSTERS_PROBAS:1){
    mtmp[mmm[[i]] <= segments_probas[j]] <- j
  }
  mmm2[[i]] <- mtmp
}

print("--- Definition des sequences pour l'apprentissage superieur")
seqdef.mmm <- vector("list", N_TEAM*N_STATES)
for(i in 1:(N_TEAM*N_STATES)){
  seqdef.mmm[[i]] <- seqdef(
    data = mmm2[[i]],
    alphabet = 1:N_CLUSTERS_PROBAS
  )
}

# Hmm sup unique
megahmm.test <- transfer_data(seqdef.mmm, fmegahmm$model)
# ================================================================================================
a <- matrice.global.test[[1]][1:10,]
b <- matrice.global.test[[2]][1:10,]
t <- sum(!is.na(a[1,]))
run <- vector("list", t)
for(i in 1:t){run[[i]] <- matrix(0, nrow=15, ncol = 15)}
for(i in 1:t){
  for(j in 1:5){
    run[[i]][a[j,i]+1,b[j,i]+1] <- 1
  }
  for(j in 6:10){
    run[[i]][a[j,i]+1,b[j,i]+1] <- -1
  }
}
for(i in 1:length(run)){
  myImagePlot(run[[i]])
  Sys.sleep(1)
}













