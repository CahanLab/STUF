# Spatial Transcriptomics Useful Functions (STUF)
Using exising Spatial Transcriptomics (ST) analysis packages to perform simple analyses is surprisingly hard. This is because most ST tools were designed to solve complex problems. On the other hand, __STUF was designed to make conceptually simple tasks easy__. For example, using STUF and just a few lines of Python you can:

<div style="
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  grid-template-rows: repeat(2, auto);
  gap: 1rem;
" markdown="1">

<div class="thin-outline" markdown="1">
  Find genes with expression patterns similar to a gene that you specify

  ---

  <figure><img src="img/similar.png" style="width:100%; height:auto;">
    <!--<figcaption><b>Find genes with expression patterns similar to a gene that you specify</b></figcaption>-->
  </figure>
</div>


<div class="thin-outline" markdown="1">
  Show the expression of two genes or gene sets on one embedding

  ---

  <figure><img src="img/bivariate.png" style="width:100%; height:auto;">
    <!--<figcaption><b>Show the expression of two genes or gene sets on one embedding</b></figcaption>-->
  </figure>
</div>



<div class="thin-outline" markdown="1">
  Rotate or flip sections

  ---

  <figure><img src="img/transform.png" style="width:100%; height:auto;">
    <!--<figcaption><b>Rotate or flip sections</b></figcaption>-->
  </figure>
</div>


<div class="thin-outline" markdown="1">
  Define gradients or contours of expression

  ---

  <figure><img src="img/contourize.png" style="width:100%; height:auto;">
    <!--<figcaption><b>Define regions based on gradients of expression (i.e. contours)</b></figcaption>-->
  </figure>
</div>

</div>