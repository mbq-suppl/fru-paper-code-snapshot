source('cranise.R')
list.files('cran/build/fru/',recursive=TRUE,full=TRUE)->p
gsub("fru","fruxt",p)->q

sort(unique(dirname(q)))->qd
print(qd)

for(e in qd){
  print(e)
  dir.create(e,recursive=TRUE)
}

for(e in 1:length(p)){
  readLines(p[e])->fc
  gsub('fru','fruxt',fc)->fcc
  writeLines(fcc,q[e])
}
