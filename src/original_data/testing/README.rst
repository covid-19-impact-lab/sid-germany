Data on testing for COVID-19
============================

The data on testing for COVID-19 is from the RKI which collects voluntary reports from
all test laboratories in Germany.

The weekly update file can be found under
https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/nCoV_node.html under "Daten
zum Download" and "Tabellen zu Testzahlen, Testkapazitäten und Probenrückstau ...".

Explanations
------------

Probenrückstau
    It is the number of tests which is still waiting to be processed which should be
    comparable to our ``"pending_test"`` column in states.


``detected_and_undetected_infections`` is scraped from
https://covid19.dunkelzifferradar.de/ (last access: 2021-01-16).
