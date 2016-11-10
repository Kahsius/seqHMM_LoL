# Script de partitionnement des données
learn <- 0.9 # proportion des données en apprentissage
test <- 1-learn

load("./package_data_lol_NoTMAX.RData")
N_P_PER_TEAM <- 5
N_TEAM <- 2
len <- dim(matrice.global[[1]])[1]
n_p_per_game <- N_P_PER_TEAM*N_TEAM
n_matches <- len/n_p_per_game

n_matches_test <- ceiling(n_matches*test)
matrice.global.test <- vector("list", length(matrice.global))
for(i in 1:length(matrice.global)){
  matrice.global.test[[i]] <- matrix(NA, nrow = n_p_per_game*n_matches_test, ncol = dim(matrice.global[[i]])[2])
}
index <- sort(sample(1:n_matches, size = n_matches_test))
for(i in n_matches:1){
  if(i %in% index){
    range.global <- 1:n_p_per_game+(i-1)*n_p_per_game
    range.test <- 1:n_p_per_game+(which(index==i)-1)*n_p_per_game
    for(j in 1: length(matrice.global)){
      matrice.global.test[[j]][range.test,] <- matrice.global[[j]][range.global,]
      matrice.global[[j]] <- matrice.global[[j]][-range.global,]
    }
  }
}
