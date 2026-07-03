"""
facts.py — a curated corpus of stable, well-established F1 facts.

RAG grounds the model in retrieved text, so the corpus has to be *accurate*. This
file holds evergreen facts that don't drift year to year — championship records,
legendary drivers, iconic circuits, and the rules/technology. Current-season data
(who drives for whom right now, latest results) comes separately from the OpenF1
API in ``openf1.py`` so this file never goes stale.

Each entry is a short, self-contained document (one idea) — the unit the
retriever embeds and returns. Kept deliberately factual and conservative; where a
number could drift (active drivers' title counts) we phrase it safely.
"""

from __future__ import annotations

# Each string is one retrievable "document". Short and self-contained on purpose.
CURATED_FACTS: list[str] = [
    # ----- Championship records -----
    "Michael Schumacher won seven Formula 1 World Drivers' Championships (1994, 1995, and five in a row from 2000 to 2004), most of them with Ferrari. He holds a share of the record for most titles.",
    "Lewis Hamilton has won seven Formula 1 World Drivers' Championships, tying Michael Schumacher's record. He holds the records for most race wins and most pole positions, and has driven for McLaren, Mercedes and, from 2025, Ferrari.",
    "Juan Manuel Fangio won five World Championships in the 1950s, a record that stood for decades and is still regarded as one of the greatest achievements in the sport.",
    "Alain Prost, nicknamed 'The Professor' for his calculated, tyre-savvy style, won four World Championships.",
    "Sebastian Vettel won four consecutive World Championships from 2010 to 2013 with Red Bull Racing.",
    "Max Verstappen is a multiple-time World Champion with Red Bull Racing; his first title came in 2021 after a dramatic season-long fight with Lewis Hamilton.",
    "Ayrton Senna won three World Championships and is regarded as one of the greatest drivers ever, especially in the wet and over a single qualifying lap. He drove for McLaren and died in a crash at Imola in 1994.",
    "Fernando Alonso won two World Championships, in 2005 and 2006 with Renault, ending Ferrari and Michael Schumacher's dominance.",
    "Nico Rosberg won the 2016 World Championship for Mercedes and retired from Formula 1 just days later.",
    "Kimi Raikkonen, nicknamed 'The Iceman', won the 2007 World Championship with Ferrari and holds the record for the most Grand Prix starts.",

    # ----- Legendary drivers -----
    "Ayrton Senna and Alain Prost had one of the fiercest rivalries in F1 history as McLaren teammates in the late 1980s, their contrasting styles — Senna's raw speed versus Prost's precision — defining an era.",
    "Jim Clark, a two-time champion in the 1960s, is considered one of the most naturally gifted drivers ever.",
    "Jackie Stewart won three championships and became a leading campaigner for driver safety in F1.",
    "Niki Lauda won three titles and famously returned to racing just weeks after a near-fatal fiery crash at the Nurburgring in 1976.",
    "Lewis Hamilton became the first and, for a long time, only Black driver in Formula 1, and a prominent advocate for diversity in the sport.",

    # ----- Teams / constructors -----
    "Ferrari (Scuderia Ferrari) is the oldest and most successful team in Formula 1 history, holding the records for most Constructors' Championships and race wins. It is based in Maranello, Italy, and its fans are known as the tifosi.",
    "Mercedes dominated the turbo-hybrid era, winning eight consecutive Constructors' Championships from 2014 to 2021.",
    "Red Bull Racing became a dominant force in the early 2010s with Sebastian Vettel and again in the early 2020s with Max Verstappen.",
    "McLaren is one of the most successful and historic teams in F1, with multiple Constructors' and Drivers' Championships across several eras.",
    "Williams is a historic, championship-winning team that was a front-running force particularly in the 1980s and 1990s.",

    # ----- Iconic circuits -----
    "The Monaco Grand Prix, run on the tight street circuit of Monte Carlo, is the crown jewel of Formula 1. Overtaking is extremely difficult, so qualifying and track position are decisive, and the barriers punish any mistake.",
    "Monza in Italy is known as the 'Temple of Speed' for its long straights and low-downforce, high-top-speed layout, with passionate Ferrari tifosi filling the grandstands.",
    "Spa-Francorchamps in Belgium is a long, fast, old-school circuit famous for the Eau Rouge-Raidillon uphill sequence and for unpredictable weather that can differ across the lap.",
    "Silverstone in Britain hosted the very first Formula 1 World Championship race in 1950 and is famous for its fast, flowing corners like the Maggotts-Becketts complex.",
    "Suzuka in Japan is a beloved drivers' circuit with a unique figure-of-eight layout and the challenging Esses, and has decided several championships.",

    # ----- Rules & technology -----
    "DRS (Drag Reduction System) is a driver-operated flap in the rear wing that reduces aerodynamic drag to boost straight-line speed and aid overtaking. It can only be used in designated zones when a driver is within one second of the car ahead, and it was introduced in 2011.",
    "ERS (Energy Recovery System) is the hybrid part of the power unit. It harvests energy under braking and from exhaust heat, stores it in a battery, and redeploys it as extra power. F1 moved to these turbo-hybrid power units in 2014.",
    "Ground effect — generating downforce from a shaped underfloor rather than mostly the wings — was reintroduced with the 2022 regulations to let cars follow each other more closely and improve overtaking.",
    "Formula 1 points are scored by the top ten finishers on a sliding scale, with 25 points for a win down to 1 point for tenth place. Points count toward both the Drivers' and Constructors' Championships.",
    "In a dry Formula 1 race, drivers must use at least two different slick tyre compounds, which forces at least one pit stop and shapes race strategy.",
    "Pirelli is Formula 1's sole tyre supplier, providing dry compounds labelled soft, medium and hard (plus intermediate and full wet rain tyres), trading grip against durability.",
    "The undercut is a strategy where a driver pits earlier than a rival for fresh tyres and uses the extra grip to set fast laps and jump ahead when the rival later stops. The overcut is the opposite — staying out longer on older tyres.",
    "A Safety Car neutralises a race by bunching the field behind it, which makes pit stops cheaper in relative time and often scrambles the order. A Virtual Safety Car (VSC) instead makes every driver slow to a set delta without a physical car on track.",
    "Formula 1 introduced a cost cap (budget cap) in 2021 that limits how much teams can spend on car performance, aiming to level the field and reward efficiency over sheer spending.",
    "A Formula 1 race weekend usually consists of practice sessions, a knockout qualifying (Q1, Q2, Q3) that sets the grid, and the race on Sunday. Some weekends also include a shorter sprint race.",

    "The halo is a titanium head-protection structure above the cockpit, mandatory since 2018. It has been credited with saving several drivers from serious injury, including Romain Grosjean in his fiery 2020 Bahrain crash.",
    "Refuelling during Formula 1 races has been banned since 2010; cars start with all the fuel they need, so pit stops are only for tyres (and repairs), typically taking two to three seconds.",
    "Parc fermé rules lock a car's setup from the start of qualifying until the race, so teams cannot rebuild the car between the two; breaking parc fermé means starting from the pit lane.",
    "Formula 1 qualifying is a three-stage knockout: Q1 eliminates the slowest five, Q2 the next five, and Q3 is a ten-car shootout for pole position — the first place on the starting grid.",
    "A sprint is a shorter ~100 km Saturday race with no mandatory pit stop, held at selected Grands Prix. It awards points to the top eight (8 down to 1) and its result sets nothing for Sunday under the current format — the grid comes from regular qualifying.",
    "Race control uses flags: yellow for danger (no overtaking), a red flag stops the session, blue tells a lapped car to let leaders through, black disqualifies a driver, and the chequered flag ends the race — 'getting the chequered flag' means winning or finishing.",
    "Grid penalties are handed out for exceeding power-unit component allocations or for driving infringements; stewards can also give time penalties (5 or 10 seconds), drive-throughs, or stop-go penalties served in the pits.",
    "The 107% rule requires a driver in Q1 to lap within 107% of the fastest Q1 time to be allowed to race, keeping dangerously slow cars off the grid.",
    "The formation lap is the slow lap before the start on which drivers warm tyres and brakes and take their grid slots; jumping the start earns a penalty, detected by sensors in each grid box.",
    "A modern Formula 1 power unit is a 1.6-litre V6 turbo-hybrid producing around 1,000 horsepower combined; cars weigh about 800 kg, pull over 5g in fast corners, and top ~350 km/h with DRS open.",
    "An F1 steering wheel is a computer with dozens of controls — brake balance, differential, engine modes, ERS deployment, radio, DRS — and drivers adjust these settings every lap while racing.",
    "F1 race distance is the fewest laps exceeding 305 km (Monaco excepted at ~260 km); races are capped at two hours of green-flag running (three including stoppages).",
    "Blue-riband records in F1: most wins in a season, youngest champion (Sebastian Vettel, 23, in 2010), and most consecutive titles (Michael Schumacher's five, 2000-2004; matched in length by no one since).",
    "Slipstreaming (towing) is following a car closely on a straight to sit in its low-drag wake and gain top speed — the classic overtaking aid alongside DRS; in corners the same wake becomes performance-sapping dirty air.",

    # ----- General / history -----
    "The Formula 1 World Championship began in 1950. It crowns two champions each year: the World Drivers' Championship and the World Constructors' Championship for teams.",
    "The 2021 season featured a dramatic title fight between Lewis Hamilton and Max Verstappen that went down to the final lap of the final race, and is remembered as one of the most intense championship battles in the sport's history.",
    "Dirty air is the turbulent wake behind an F1 car that reduces the downforce of a following car, making it slide and overheat its tyres — a key reason overtaking can be difficult and why DRS and slipstreams matter.",
]


def curated_documents() -> list[dict[str, str]]:
    """Return the curated facts as retriever documents with a source tag."""
    return [{"text": fact, "source": "curated"} for fact in CURATED_FACTS]
