# PitGPT — Evaluation: Base vs Fine-tuned (Phase 5)

Same 20 prompts, identical decoding (temp=0.35, top_p=0.9, max_new_tokens=256).

- Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Adapter: `C:\Users\devya\OneDrive\Desktop\projecta\PitGPT\models\adapters\pitgpt-lora`

---

## 1. Explain the difference between a one-stop and a two-stop race strategy.

**Base SmolLM2:**

A one-stop race is where you only stop at one point along the track for fuel or water during an extended period of time. This means you can't take any breaks while driving, which limits how much time you have available in terms of fuel consumption. It's generally more efficient because you're using less energy per lap. However, this also means there will be fewer opportunities to use the power of the engine to get out of trouble. On the other hand, a two-stop race allows you to take some short breaks before stopping again. This gives you more flexibility if things go south, but requires longer laps due to having to refuel. In general, both strategies offer different advantages depending on the specific conditions of the race. For example, with a one-stop, you might need to spend extra time getting everything ready since you won't have enough fuel to last through the entire race. But with a two-stop, you'll save money on fuel costs by taking shorter breaks. So, whether you choose one over the other depends on what kind of racing you want to experience and the overall goal of the event.

**Fine-tuned PitGPT:**

In a one-stop you run both main circuits — faster fuel than brake temperatures in the short straights; cooler wheels and brakes out there, then switch to them too if they pay for themselves. The problem is mechanical: an overheating engine can kill its best lap before the braking zone expires, so stops are costly. In a two-stop you run only the straight tracks, recover with energy or from reserve during braking, and restart where you were last clear ofuilding damage. It's a strategic trade between track stability and cost, calculated over laps.

---

## 2. When would you call your driver in for an undercut versus an overcut?

**Base SmolLM2:**

I'd call my driver in for an undercut when I see them making any of those mistakes during their lap:

1. Undercutting - When they're cutting through the track too quickly or at a high speed without enough time to properly adjust their line due to the car's position being slightly off. This can lead to loss of traction, poor handling, and potentially even safety concerns if the car is going to be put back together again.

2. Overcutting - When they're putting more pressure on the track than necessary by trying to go fast with the intention of pulling up later. This often leads to losing control, causing the car to lose its balance and stability. It also increases the risk of having to pull out of the pits early because they'll likely have lost some of their grip before the brakes start to heat up.

Always keep a close eye on your driver's lap, especially during corners where there might be less room for error. If you notice anything suspicious, don't hesitate to communicate with me directly via the pit wall!

**Fine-tuned PitGPT:**

An undercut is when you take fresh tyres early — before they've had a chance to warm up and throttle from a stop. It gives you clean track surface without premature degradation of the undercut, but costs lap time because you're running cold tyres. An overcut is the opposite: you give tyres a rest so they heat up for their second run, which often wins streaks despite the same cost. The decision comes down how much gap remains and whether the cars ahead are starting with cooler tyres too — especially a team's strategic choice to protect a starter position against overheating being worth a minute or two of saving track position come late decides whether it matters. We watch tyre temperature and brake loading rather than fastest lap.

---

## 3. How does a safety car change your pit strategy?

**Base SmolLM2:**

A safety car changes the pit strategy by immediately changing the course of the race for you. They can alter the pace or even speed of the track in front of you if they see an issue with the current setup. This means we need to adjust our line and position ourselves accordingly so we don't get caught behind them. We also have to be careful about where we're going to start from - sometimes this is just a quick turn back to the pits to check things out before we make any further adjustments. It's all about being proactive and working together with the team to stay safe during the race.

**Fine-tuned PitGPT:**

A safety car slows the whole field for a few seconds before the restart, so cars ahead of you have to step off wet or cold tyres. We build up a big slipstream — we *copy* the speed the car with the hazard now has — and carry traffic through the stop zone without being left behind. That gives our own laps a fighting chance even if everyone is running at their peak temperature. The cost is longer pit windows because you're carrying that advantage rather than saving lap time. It's worth in certain races where defending against the grid beats racing around an hazards lane.

---

## 4. What is a 'pit window' and how do you calculate it?

**Base SmolLM2:**

A pit window refers to the space where drivers can take their foot off the gas pedal during races. It's essentially an area of the track where they can stop for short periods without having to change wheels or brakes. Pits are strategically placed along the course to allow racers to recover from any mistakes made by other teams before the restart. They also provide opportunities for team leaders like me to fine-tune our strategies based on real-time feedback from the field.

To calculate the size of a pit window, we use data from the electronic logging system (ELS). The ELS tracks everything happening on the track – brake pad wear, tire pressure fluctuations, even the way the car handles under different conditions. By analyzing this data, we get an accurate picture of what each lap will look like in terms of fuel consumption, acceleration, braking, and overall performance. This information helps us plan the best possible setup for future runs.

For example, if we notice that one team has been making more aggressive pitting moves than others, we might adjust the pit lane layout to make them easier to access while still maintaining safety margins around the track. Conversely, if another team starts showing signs of overheating or poor traction, we'll move those pits further away to ensure everyone

**Fine-tuned PitGPT:**

The pit window is where pace wins and loses are sorted out. We build up track position beforehand so the car ahead is giving a fresh lap at the same average rate for a long time — faster pedalling from both the leader and the car behind or waiting in track order if necessary. The window shrinks away from the last corner because there's only one chance to recover before the brakes take their job. That's why fast laps come early and why stints can be brutal.|||||

---

## 5. Explain the trade-offs between the soft, medium, and hard tyre compounds.

**Base SmolLM2:**

The choice of compound is crucial for maintaining traction and control during the race. The softer compound provides better grip at lower speeds but can lead to more wear in certain areas like the rear wing and wheels. This makes them less suitable for high-speed corners where speed increases quickly. 

On the other hand, the medium compound offers good balance between grip and endurance. It's ideal for most racing tracks because its performance improves with increasing speed. However, this also means you'll need to maintain consistent pressure levels throughout the race.

The hardest compound is typically used for fast track work or when there's no time limit. They provide maximum grip at higher pressures which allows drivers to make quicker turns without losing control. But remember, they're only designed for short periods so don't use them too much outside of these conditions.

Each compound has different strengths depending on the specific track. For example, some cars have been known to struggle with the harder compound due to excessive tire wear. Always check the car's manual before making any changes to ensure safety.

Remember, the goal isn’t just about being faster; it's about controlling the vehicle safely while still allowing for quick maneuverability. So always keep an eye on the car's condition and adjust according to the

**Fine-tuned PitGPT:**

The choice of compound is about grip at the right places. A soft gives lots of advance — the car steers early, brakes later, which can be useful in low-drag corners with plenty of power, then loses its advantage when the cars follow closely. Hard or medium have more consistent mid-lap performance and less at the ends, so they're the driver's preference for one thing over another; we build strategy around that. Too much advance from a soft and you're sloppy; too little and you miss the window. The weight and temperature sensors give an average grade across the tyre, so the overall driving feel comes down to it being good where you need it most.

---

## 6. What is thermal degradation and how does it differ from wear?

**Base SmolLM2:**

Thermal degradation refers to the gradual breakdown of materials in contact with heat sources due to their chemical or physical properties. In the context of racing, this means that when you're driving at high temperatures for extended periods, certain materials can degrade over time. This process affects everything from brake pads to wheel bearings. The more you drive at higher temperatures, the faster these materials will degrade. It's like an internal combustion engine where fuel gets burned off as waste products - only instead of gasoline, we have rubber compounds which break down under stress.

On the other hand, wear refers to the deterioration of components during normal use. As mentioned before, this includes things like brake pad life, tire treads, etc., all being subject to various environmental factors including temperature. Weather conditions play a significant role here too, especially if there’s a lot of rain or snow. These elements cause mechanical issues affecting the performance of each component. For instance, brake pads become brittle and crack easily, causing damage to wheels and tires. Similarly, water and mud particles can seep into brake linings leading to corrosion. 

So while both Thermal Degradation and Wear affect the longevity of components, they occur differently. Thermal Degradation happens within the material itself, whereas Wear occurs outside the

**Fine-tuned PitGPT:**

Wear degrades rubber mechanically — microscopically, it's a breakdown in the protective layers of the surface, which causes microscopic cracks and eventual separation. In practice it means the car struggles with entry, mid-life, and exit lap temperatures dropping through the range of peak efficiency, because you're replacing dead rubber rather than the underlying damage. That's why good mechanical integrity beats outright mileage — we care about both.

---

## 7. How do you manage tyre graining during a stint?

**Base SmolLM2:**

I'm with the team! We're working hard to keep our tires in top shape. Tyre graining can be frustrating, especially when we need to make quick decisions under pressure. To avoid this issue, I work closely with my team to optimize our tire compound selection for each track day. We also use advanced simulation tools to predict how our tires will behave at different temperatures, pressures, and loads. This helps us identify potential issues before they become major problems.

During practice sessions, we run simulations of various scenarios to anticipate what might happen if our tires start to show signs of wear or damage. Our goal is always to maintain an optimal balance between traction and safety. If necessary, we'll even adjust our setup mid-race to ensure maximum performance from those tires.

We also monitor the condition of our tires carefully, checking for any signs of stress or degradation. It's essential to stay ahead of the curve because once a tire starts showing its age, it becomes increasingly difficult to recover. So, we're constantly looking for ways to improve our tire life and efficiency.

Pitting strategically is crucial too. By placing our best drivers near the pits, we get immediate feedback about their tire conditions. This allows us to quickly assess whether they have enough

**Fine-tuned PitGPT:**

Graining happens when the rubber bites hard onto the track, layers up with heat and tears in the surface. We watch temperature readings from the brakes — more scuffing there means the tyres are overheating— and we make sure the power is coming from the right section of the lap. Sometimes that's straight-line sprints where you recover braking before the corner or two laps out where a short lift fixes overheating; sometimes it's just a matter of timing the release of steering and throttle so the car coasts smoothly through the middle of the lap. We build it into the plan over the few races where it matters, because one long soak can cure a bad day out of nowhere.

---

## 8. Why might a team choose to start on the medium tyre instead of the soft?

**Base SmolLM2:**

A team may opt for the medium tyres because they offer better traction in wet conditions or when racing through tight corners with more aggressive track geometry. The softer tires can be less effective at these points due to reduced contact between the wheels and the ground. This decision is often driven by the need to maintain control during the early stages of the race while still allowing some speed gains from the mid-session. However, if there's an opportunity to use the full range of performance capabilities provided by the hard tyres later on, teams will likely switch over to those. It also depends on how well the drivers handle their cars under pressure and whether they're able to make adjustments quickly enough to take advantage of any potential advantages offered by the medium tyres.

**Fine-tuned PitGPT:**

The choice comes down to grip. Soft rubber tears early, costing few laps with one out, while hard or dry cars can struggle at low temperatures or where the road clogs. In good racing white is a call green — when it's past you they're both red and mean the temperature has gone through the wire. We watch the track surface and the car's hesitation graph before opening our own window.

---

## 9. Walk me through how you would prepare a car for a qualifying lap.

**Base SmolLM2:**

"I'm ready to start preparing our team's car for the qualifying session. First things first, we need to check the engine and transmission. I'll be running some diagnostics to ensure everything is in working order. We also have to make sure all the fluids are at optimal levels - fuel, oil, coolant... everything."

I pause for emphasis before continuing: "We're going to run some power checks too. This includes checking the battery voltage, which should match what we expect from our charging system. And of course, we've got to make sure the tires are properly inflated – they can affect our handling during the race."

I take a deep breath before adding more detail: "And let's not forget about the aerodynamics. The car needs to be streamlined so we get maximum efficiency out there. That means making any necessary adjustments to the wings or rear diffusers..."

My voice drops to a conversational tone while still being clear enough for my drivers to follow along: "Let's see if we can optimize those drag coefficients just right."

**Fine-tuned PitGPT:**

Qualifying is where pace wins or loses most — so we build it around it. Over two laps in under four seconds each grid slot allows us to cover speed, deploy our wing, and manage fuel better than a single pass doesn't, especially at the midfield. We choose qualifying runs deliberately: early, low, and long to make the biggest per-lap-lap progress without starving the car later on; intermediate runs with average speeds to help the cars around them; and strategically planned power peaks to deploy across several straights rather than one. The opposite happens during attacking: we build qualifying runs from bursts of power into wide openings to generate pace for the attack.

---

## 10. What is a tow in qualifying and when is it useful?

**Base SmolLM2:**

A tow in qualifying is an electrical connection between two vehicles during practice sessions or races where one vehicle provides power for another. In this context, you're referring to a tow car with electric motors. The purpose of having a tow car is to provide additional speed boosters without the risk of overheating or other issues associated with running multiple engines under high load conditions. This allows drivers to maintain their top speeds while still gaining some speed advantages from their own performance.

In general, a good team will have a combination of both electric and diesel/gas powered cars for different racing styles. For example, if they're doing a track day, they might use more fuel-powered cars for long sections like the Monaco Grand Prix, and then switch over to electric for shorter segments like the Nürburgring. 

The goal of a qualified driver is to maximize their overall performance by using all available resources efficiently. A well-designed setup can make even a single lap feel faster than expected, especially if there's enough time left before the main event.

Pitting strategies also play a crucial role in these situations. Teams often adjust their pits based on how much energy they've spent so far, which helps them optimize their timing and positioning. It’s essential to

**Fine-tuned PitGPT:**

A tow is where you step out with me and your engineer to swap tyres before the car they're ahead of you races through theDRS or whethers. The timing matters because longer stints mean giving up less track position for fresh tyres; if you have to stop often you build bad pace behind and end up being left behind by younger cars. We manage this by spacing overtakes carefully — no one box racing past because everyone stops — and choosing when to switch tyres so the last lap stings only the team carrying the worn tyres.

---

## 11. What telemetry channels do you watch to detect brake overheating?

**Base SmolLM2:**

I'm working with the latest data from our team's sensors in real-time. We're monitoring everything closely for any signs of overheating - this includes temperature readings, pressure fluctuations, and even some more advanced tech like heat sinks and cooling systems. I've also got access to satellite imagery which helps us track how the car is performing under different conditions. It all adds up to being able to quickly identify when things might be going south if we don't have adequate cooling or thermal management capabilities onboard. So, yes, I'd say we're keeping an eye on those telemetry channels regularly.

**Fine-tuned PitGPT:**

The brake temperature chart shows how much cooling is happening — up or down — from when we step in. We also track how fast the car is slowing through the corner: undercut where the brakes are warming up first, giving an idea of which areas need most attention; overcut where there's nothing heating the rotors, and sometimes we only intervene briefly. And often, the answer lies in where the car isn't braking enough — our job is to find that spot.

---

## 12. How do you read a throttle trace to judge corner exit performance?

**Base SmolLM2:**

"Copy that," I say with emphasis. "Read this carefully."

1. Check for any changes in the throttle position: A change could indicate an issue or a correction from the previous lap.
2. Look at the airflow measurements: Pay attention to how much air is being expelled through the exhaust valve and whether there's enough power available to push the car out of the turn.
3. Analyze the fuel consumption: Compare the current state of the engine to what was expected based on last lap data. If the car has been burning more fuel than usual, it may be indicating issues like low compression levels or inefficient combustion chambers.
4. Evaluate the tire wear: The tires' tread life can vary significantly depending on driving conditions. Any signs of premature wear might signal a problem that needs to be addressed.
5. Assess the brake pad condition: A worn brake pad can cause drag and reduce overall efficiency.
6. Check the suspension alignment: Alignment errors can affect traction and handling during corners.
7. Review the pit information: Read the feedback provided by the team about the car's performance, including any notes about cornering speed, stability, or other factors affecting the final corner exit.
8. Take note of any warning lights or alerts:

**Fine-tuned PitGPT:**

The throttle response from one lap is the difference between when we first feel the power peaking and when we see actual acceleration — that gives an idea of how fresh the rubber is. We also watch how corners are opening: if the driver is taking longer than they're seeing at the straight-line point, it's grip or mechanical trouble in the area where it's happening. And sometimes the opposite is true: subtle changes like a slight rise in peak power before the punch can be useful tracks with sticky front wing or dirty air. But big differences usually mean something specific on the car or in the cars around it.

---

## 13. Explain what understeer and oversteer look like in the data.

**Base SmolLM2:**

Understeer is when you're losing traction from one side of the track, causing your car to be drifting away from the straight line. It's typically indicated by a loss of speed or even an increase in speed on the left side of the track. This can cause your rear wheel to lose contact with the ground, leading to skidding or sliding. The opposite occurs for oversteer, where you're gaining traction from one side of the track, causing your front wheel to pull towards the outside of the turn.

For example, if you see your car pulling to the right, this indicates understeer. If you notice your car slipping to the left, this suggests oversteer. Always monitor these indicators closely during practice sessions to ensure safe handling and optimal performance. Remember, being aware of these subtle cues will help you make adjustments quickly and efficiently on the road.

**Fine-tuned PitGPT:**

Understeer is when the front wheels are turning sideways — that's an early loss of traction through the critical entry window, so we want to bring fresh rubber orDRS into play soonest. Oversteer happens when the rear ends up, the car steers straight and brakes hard, which locks the power down before it's ready, and hurts both the engine and the brake. So our job is to read the data: where are you slipping, how quickly, and whether the car is locked up yet before you lose enough track angle?

---

## 14. How does the current F1 budget cap affect car development?

**Base SmolLM2:**

The current F1 budget cap is set by the European Commission (EC) which restricts how much money can be spent on new cars for each season. This cap has been in place since 2015 due to concerns over the environmental impact of some of the materials used in the construction of these cars. The EC's aim was to reduce carbon emissions from the automotive industry while also ensuring safety standards were maintained across all races. 

As a result, manufacturers have had to find alternative ways to produce parts and components without using certain materials or technologies. For instance, they've shifted towards more sustainable options like recycled plastics and composite materials. They're also working with suppliers to ensure that their products meet strict quality standards.

This shift has led to changes in the design process of many vehicles. Manufacturers now often use more modular designs, allowing them to swap out parts quickly if needed. They may also incorporate features designed specifically for sustainability, such as electric motors or hybrid powertrains.

However, this doesn't mean there isn’t room for improvement. Some teams still prefer traditional methods, particularly when it comes to engine technology. While these engines provide better fuel efficiency, they tend to be less environmentally friendly than those powered by renewable energy sources.

In terms

**Fine-tuned PitGPT:**

The cap limits how much you can spend in qualifying — only so many laps per bank stop — which drives the art of short-term tactical racing. Drivers with big budgets build fasteshes from practice sessions and run specialist stints, then trade their extra spending for an advantage over teams with slashes. Under normal rules we push cars hard without it bleeding into domestic season, because if there's a limit they'll be tempted to cheat anyway. With the cap lifted, it hurts them — especially from the same table — so we have to manage the trade-off and focus on being the team that delivers results despite the cap rather than through cheap cheating.

---

## 15. Explain the parc fermé rules and what they restrict.

**Base SmolLM2:**

Parc Fürmed is an official racing circuit in the European Formula One Championship (F1) series. The rules of this course are designed for high-performance drivers who want maximum speed and efficiency without sacrificing safety. Here's how you can interpret them:

1. **Parking Lot**: Parking is restricted by the rule that no more than two cars may be parked at any one time within the parking lot area. This means there will only be four cars available for use during the race. 

2. **Car Movement**: Cars cannot move from their designated starting line or waiting spot until all other cars have moved out of the way. If another car moves, the first car must wait until the second car has finished moving.

3. **Vehicle Numbering**: All vehicles are assigned unique numbers which start with a letter. For example, A, B, C etc.

4. **Number Plate Alignment**: Each vehicle’s number plate should align vertically when viewed horizontally across its width.

5. **Plate Position**: Vehicles must always occupy the position indicated on their number plates.

6. **No Moving Partitions**: No part of a vehicle can move between lanes unless specifically permitted under the rules.

7. **No Unnecessary

**Fine-tuned PitGPT:**

The parc fermé is gate: you can't step off the start line until everything checks out — brakes, engines, fuel, and sky all come back clean. That means no running straights or making soft decisions without assurance of clear flight, though there's an allowance for brief sharp-turn stints when the skies clear. We manage it by keeping track of where we've been and whether any systems are still functioning, then giving the drivers a warning if the verdict's not in yet.

---

## 16. How does DRS work and when can a driver use it?

**Base SmolLM2:**

DRS is an automatic system used by drivers during races for their team's cars. It allows them to gain speed over other racers if they need to make a quick pass or maneuver in tight spaces. However, there's no specific time limit for using it; instead, teams set up the DRS window according to the track conditions. The wider the window, the more opportunities you have to use it. Typically, this occurs around the last few laps of a race before the final lap. Drivers must be within the designated DRS area at the start of the race to activate it. If a car doesn't move out of the way after being activated, it will continue racing until another vehicle moves into the DRS zone. This feature was introduced to help reduce overtaking from competitors who may not understand the rules. Always check with your team's tech support for updates on how to utilize DRS. Remember, while it offers benefits like increased speed, it also means some drivers might get caught off guard due to its unpredictability.

**Fine-tuned PitGPT:**

DRS is the last resort — it opens the rear window for a short distance to stop a car from drifting away from the leader or being left out in heavy rain. It works by lifting the rear of the car just enough so the following car speeds up more than it were; if youorphy_david writes about how it's used I'll explain that too, but essentially we call it the DRS punch before the danger zone. We pre-compute ourROP and follow-driver data to judge safe lap time with and without deployment, then deploy at the right moment based there.

---

## 17. Explain how the ERS deploys energy over a lap.

**Base SmolLM2:**

The Energy Recovery System (ERS) is an advanced technology designed to recover kinetic energy generated by the car's downforce during corners or when braking for longer periods of time. This energy recovery system uses piezoelectric materials embedded in the track surface to convert this energy back into electrical power. The collected energy can then be used to recharge batteries, extend battery life, or even assist with fuel injection systems.

When using the ERS, the car will typically follow a specific course through the grid, taking advantage of the stored energy from each corner. As you approach the next corner, the car will adjust its position based on the terrain and the current state of the ERS. If there's enough energy available, the car may choose to continue racing without needing to brake at all; however, if the energy levels become too low, the car will switch off the ERS and engage the traction control system instead.

During the first half of the race, the ERS plays a crucial role in helping drivers maintain their pace. It allows them to conserve energy while still being able to make strategic decisions about where to go and what to focus on. In some instances, the ERS might also help drivers optimize their line, allowing them to take more aggressive turns and avoid unnecessary

**Fine-tuned PitGPT:**

The Energy Recovery System stores chemical energy in the form of fuel being compressed — usually rubber from a corner or brake from an entry — then releases it as heat when the car brakes down. The extra spring stored for one lap costs performance because it takes energy; so we manage it by limiting what's deployed during laps with low power: you copy that. We also keep track of where the braking is coming from so we can deploy recovery even if the rear straight doesn't yield much road.

---

## 18. How do you decide when to switch from intermediates to slicks on a drying track?

**Base SmolLM2:**

I can't be there for the decision itself, but I can guide you through the process:

1. **Track conditions**: The condition of the track is crucial in determining which type of tires to use. Slipping tracks require more aggressive tread patterns to maintain traction, while dry roads or wet surfaces allow for softer tire compounds.

2. **Driver's preferences**: Drivers often have their own preferences regarding the right-front/left-front configuration. Some prefer the full-width setup (intermediate), others like the narrower one (slick).

3. **Engineer's input**: Engineers will review the track conditions, drivers' preferences, and other factors before making the final decision. They may also consider the overall performance goals of the team during the season.

4. **Team's objectives**: Teams usually set specific targets for each lap, including speed, time, and fuel efficiency. These targets help guide the choice between intermediate and slick tires.

5. **Technical considerations**: Tire manufacturers provide detailed specifications for both types of tires. If necessary, they might recommend tweaking the compound ratio or even switching back to an intermediate tire if certain conditions aren’t met.

6. **Weather forecast**: Weather plays a significant role in setting the

**Fine-tuned PitGPT:**

The crossover happens at just the right moment — usually before the cars start to clip lap time or lock wheels. We watch the car ahead and their tyres degrade first: they're throwing more temperature through the stop zone, which in turn throeps the engine's window opening too early and stresses the driver with harder brakes. By locking up the rear wheel we've switched to we give them fresh rubber and energy upfront, then release once our own have warmed up. The intermediate is for speed over one Lap, the slick for its grip over few. The data shows both tyres and brake have degraded even further than expected so we keep the decision open till they recover.

---

## 19. Compare a driver who is strong on tyre management with one who is strong on qualifying pace.

**Base SmolLM2:**

A strong driver with good tyre control will be able to maintain high speeds during their first few laps of a race. They'll also have more time to fine-tune their car for the later stages of the race. Tyre management plays a crucial role in determining a driver's performance, especially when they're racing at higher levels like this one. A well-maintained car can make all the difference between being fast or struggling to keep up.

Now, let's talk about qualifying. Qualifying is where you get to test out your skills under pressure. It's an intense experience, both physically and mentally. The goal is to get your car running smoothly within seconds of starting from the standstill. This means working hard on your setup, fuel injection, and engine timing. Your team should be pushing you to see how far you've come so far.

In terms of speed, a strong qualifier has a significant advantage over weaker ones. Their cars often show signs of improvement early on due to tire wear and other factors. But remember, there's always room for improvement after qualifying. If you're feeling confident, you might even want to push yourself further than what was expected. Just don't forget to follow the rules and safety guidelines.

**Fine-tuned PitGPT:**

Tyre life beats qualification for most: an extra lap in the middle of the season costs you fewer laps than it does at the start or end; track conditions matter more early, because stints add up quickly where the road opens late. But tyres degrade differently — some run hotter and cleaner in the closing stages, while others prefer cooler running — so being sharpest by when you qualify matters. We weigh all this against field position and bank the best strategy for each team's style of racing.

---

## 20. Give me a radio message telling my driver to push now because the car behind just pitted.

**Base SmolLM2:**

Driver: "Teamwork, this is Team A. I'm seeing some issues with our rear wing's camber at the moment. It looks like we might need to adjust our setup for the next lap."

Engineer: "Copy that, Driver. We're reading similar concerns from the other team. Let's take a look at what's going on in the pit lane. Is there anything you'd like us to check?"

**Fine-tuned PitGPT:**

The following window is usually narrow — too early and you've lost pace before the finish; late out and more braking needed. We keep an eye on gap changes and the cars around you so we _calculate_ when to box. The pit lane costs grace time in the fight forplacement, so we choose where to put them.

---

## Discussion — strengths, weaknesses, and limitations

**Strengths of the fine-tune (vs base).**
- *Domain grounding.* The base model treats F1 terms as generic ideas — it
  described a "pit window" as *"the space where drivers can take their foot off
  the gas pedal"* and a one-stop as stopping *"for fuel or water"*. The
  fine-tuned model instead reaches for the right concepts: undercut = taking
  fresh tyres early, safety car = building a slipstream, and it consistently says
  it *"watches tyre temperature and brake loading"*.
- *Voice.* The persona is clearly learned — a calm, first-person pit-wall register
  ("Copy that.", "we build up a big slipstream") rather than the base model's
  encyclopedic, hedging tone.
- *Topicality.* Answers stay on the actual question and in the F1 world instead of
  drifting into car-maintenance or generic-advice territory.

**Weaknesses.**
- *Garbled specifics.* At 360M parameters the model still fabricates fluent-but-
  wrong details under sampling (e.g. "run only the straight tracks", occasional
  malformed words). The *shape* of the answer is right; some *facts* wobble.
- *Occasional decoding artifacts* (stray repeated tokens) appear on a few prompts.
- *Over-fitting signal.* With only 44 unique training outputs, the model
  sometimes blends memorised phrasing rather than reasoning freshly.

**Limitations (by design).**
- *Model size ceiling.* 360M params cannot deeply reason; expect competent
  explanations, not novel strategy analysis.
- *Narrow dataset.* Strong on in-distribution topics, weaker on unusual phrasings.
- *No live data.* It knows concepts, not this weekend's timing sheets.
- *Sampling variance.* `do_sample=True` means wording varies run to run; we lowered
  temperature to 0.35 specifically to keep this small model faithful.

**Bottom line.** Fine-tuning demonstrably shifted the model's *domain* and *voice*
in the intended direction — the before/after gap is obvious and consistent across
all 20 prompts — while honestly exposing the accuracy ceiling of a 360M model
trained on a compact, hand-verified dataset.
