
# Action, Reward and State handles
There are a few options for how to pass around these objects:
    1. Pass them by value always.
    2. Pass them by pointer/reference. They are created by an Environment and stored on the heap.
    3. Pass them as IDs, where the IDs are considered unique.

Similarly, we have parallel choices with respect to their storage:
    1. Store a value.
    2. Store a reference/pointer.
    3. Store an ID.

These options differ in their effects on:
   * performance
   * error susceptibility
   * debug friendliness
   * API simplicity
   * design flixibility


### Storage

1. Store values
* Bad performance. How bad? What are some profile results? We should restrict the length of names
  in order to have meaningful discussion.

2. Store pointers/references
* Using pointers and references will make the API more complex. reference_wrappers may be needed in
  places- while they are useful, they make the code harder to read quickly.
* Using pointers will make implementing copying/moving very difficult and error prone, as any object
  that gets copied must decide weather to point to a new object that also got copied or to continue
  pointing to the original object.

3. Store IDs
* Susceptible to errors, as the ID types are interchangable. Maybe this problem can be reduced by
  compiling with flags that prevent implicit conversion. Would this even help if the type aliases
  are aliases for the same underlying type? Would the explicit keyword help us? If it did, it would
  be burdomsome to have to use explicit everywhere by default.
* Bad for debugging, as the names of actions, rewards etc must be looked up. This may seem trivial,
  but having easily displayable names can help to quickly identify a pattern which would otherwise
  go unnoticed.

For the moment, I'll try and pursue store by value. If it becomes troublesome, store by
ID is probably the next one to try.


### Argument and return types

Argument and return types don't have as many issues as storage types.

For example, the signature:

const State& next_state(const Environment& e) const;

Doesn't really have implications for copying/moving. There is no confusion as to which environment
owns the returned state. We still have the drawback of having to use std::reference_wrapper when
we need to return collections of references.

For the moment, I will go with using references and reference_wrappers.
It may be desirable to switch to using pass by value in future.

# TODO List

* Consider removing usages of Expects and Ensures in favour of using glog methods directly. Or 
  define our own macro that wraps glog. glog has better reporting on asserts.
* Implement GPI
* Sort out a propper design for transient Rewards (specifically, deal with their ids).
* Start to move code into cpp files, as the compile time is starting to get annoying.

