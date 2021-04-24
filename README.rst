sid-germany
===========

.. image:: https://img.shields.io/github/license/covid-19-impact-lab/sid-germany
   :alt: GitHub

.. image:: https://readthedocs.org/projects/sid-germany/badge/?version=latest
    :target: https://sid-germany.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/workflow/status/covid-19-impact-lab/sid-germany/Continuous%20Integration%20Workflow/main
   :target: https://github.com/covid-19-impact-lab/sid-germany/actions?query=branch%3Amain

.. image:: https://results.pre-commit.ci/badge/github/covid-19-impact-lab/sid-germany/main.svg
    :target: https://results.pre-commit.ci/latest/github/covid-19-impact-lab/sid-germany/main
    :alt: pre-commit.ci status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


This project contains the research code which applies `sid
<https://github.com/covid-19-impact-lab/sid>`_ to Germany and accompanies several
publications. See below for a list.


Usage
-----

Most data sets for this project are freely available and are added to the repository or
will be downloaded while executing the project. One data set, the microcensus of 2010,
is provided by the FDZ (Forschungsdatenzentren) for researchers for free after a
`registration <http://www.forschungsdatenzentrum.de/de/campus-files>`_. Place the the
``.dta`` of the census under
``src/original_data/population_structure/microcensus2010_cf.dta``.

To create the environment for running the code, we rely on `conda
<https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_. Type

.. code-block:: console

    $ conda env create
    $ conda activate sid-germany

to create and activate the environment. The workflow of the project is managed by
`sid-germany <https://github.com/covid-19-impact-lab/pytask>`_. Built the project with

.. code-block:: console

    $ pytask


Publications
------------

sid-germany provides the code for the following publications. Take a look at the
releases to replicate a certain publication.

- Gabler, J., Raabe, T., & Röhrl, K. (2020). `People Meet People: A Microlevel Approach
  to Predicting the Effect of Policies on the Spread of COVID-19
  <http://ftp.iza.org/dp13899.pdf>`_.

- Dorn, F., Gabler, J., von Gaudecker, H. M., Peichl, A., Raabe, T., & Röhrl, K. (2020).
  `Wenn Menschen (keine) Menschen treffen: Simulation der Auswirkungen von
  Politikmaßnahmen zur Eindämmung der zweiten Covid-19-Welle
  <https://www.ifo.de/DocDL/sd-2020-digital-15-dorn-etal-politikmassnahmen-covid-19-
  zweite-welle.pdf>`_. ifo Schnelldienst Digital, 1(15).

- Gabler, J., Raabe, T., Röhrl, K., & Gaudecker, H. M. V. (2020). `Die Bedeutung
  individuellen Verhaltens über den Jahreswechsel für die Weiterentwicklung der
  Covid-19-Pandemie in Deutschland <http://ftp.iza.org/sp99.pdf>`_ (No. 99). Institute
  of Labor Economics (IZA).

- Gabler, J., Raabe, T., Röhrl, K., & Gaudecker, H. M. V. (2021). `Der Effekt von
  Heimarbeit auf die Entwicklung der Covid-19-Pandemie in Deutschland
  <http://ftp.iza.org/sp100.pdf>`_ (No. 100). Institute of Labor Economics (IZA).
