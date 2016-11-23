extract_data_team <- function(matrice.global, team, p_per_team, n_team){
  l <- length(matrice.global)
  m <- vector("list", l)
  d <- dim(matrice.global[[1]])
  n <- d[1]/(n_team*p_per_team)
  
  for(i in 1:l){
    m[[i]] <- matrix(NA, nrow = n*p_per_team, ncol = d[2])
    for(j in 1:n){
      m[[i]][1:p_per_team + (j-1)*p_per_team,] <- matrice.global[[i]][1:p_per_team + (team-1)*p_per_team + (j-1)*(n_team*p_per_team),]
    }
  }
  
  return(m)
}