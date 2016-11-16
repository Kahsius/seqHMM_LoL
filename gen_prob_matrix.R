gen_prob_matrix <- function(row, col, triangle = FALSE) {
  m <- matrix(runif(row*col), row, col)
  if(triangle && row == col){
    m <- m*upper.tri(m,diag=TRUE)
  }
  m <- apply(m, 1, function(vector) {
    t(vector/sum(vector))
  })
  return(t(m))
}