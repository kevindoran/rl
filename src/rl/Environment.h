#pragma once

#include <string>
#include <vector>
#include <memory>
#include "util/DereferenceIterator.h"
#include <gsl/gsl>
#include "util/RangeWrapper.h"

namespace rl {

/*
 * Following the reasoning here and avoiding unsigned integer types.
 * https://google.github.io/styleguide/cppguide.html#Integer_Types
 */
using ID = int;
using Weight = double;

class State {
public:
    State(ID id, std::string name)
            : id_(id), name_(std::move(name))
    {}

    explicit State(ID id)
            : id_(id)
    {}

    const std::string& name() const {return name_;}
    ID id() const {return id_;}

    bool operator==(const State& other) const {
        return id_ == other.id_;
    }

    bool operator!=(const State& other) const {
        return !(*this == other);
    }

private:
    ID id_;
    std::string name_{};
};

class Action {
public:
    Action(ID id, std::string name)
            : id_(id), name_(std::move(name))
    {}

    explicit Action(ID id)
            : id_(id)
    {}

    const std::string& name() const {return name_;}
    ID id() const {return id_;}

    bool operator==(const Action& other) const {
        return id_ == other.id_;
    }

    bool operator!=(const Action& other) const {
        return !(*this == other);
    }

private:
    ID id_;
    std::string name_{};
};

class Reward {
public:
    Reward(ID id, std::string name, double value)
            : id_(id), name_(std::move(name)), value_(value)
    {}

    Reward(const ID id, double value)
            : id_(id), value_(value)
    {}

    ID id() const {return id_;}

    const std::string &name() const {return name_;}

    double value() const {return value_;}

    void set_value(double value) {
        value_ = value;
    }

    bool operator==(const Reward& other) const {
        return id_ == other.id_;
    }

    bool operator!=(const Reward& other) const {
        return !(*this == other);
    }

private:
    ID id_;
    std::string name_{};
    double value_;
};

/**
 * Represents a transition and its probability of occuring.
 *
 * \prob is the probability of moving from \c state to \c next_state and getting reward \c reward
 * when \c action is taken.
 */
class Transition {
public:
    Transition(
            const State& state,
            const State& next_state,
            const Action& action,
            const Reward& reward,
            const Weight prob_weight=1)
            : state_(state), next_state_(next_state), action_(action), reward_(reward),
              prob_weight_(prob_weight)
    {}

    Transition(const Transition& other) = default;
    // Delete the following constructors until needed.
    Transition& operator=(const Transition& other) = delete;
    Transition(Transition&& other) = delete;
    Transition& operator=(Transition&& other) = delete;

    const State& state() const      {return state_;}
    const State& next_state() const {return next_state_;}
    const Action& action() const    {return action_;}
    const Reward& reward() const    {return reward_;}
    Weight prob_weight() const      {return prob_weight_;}

    bool operator==(const Transition& other) const {
        return  state_       == other.state_ and
                next_state_  == other.next_state_ and
                action_      == other.action_ and
                reward_      == other.reward_ and
                prob_weight_ == other.prob_weight_;
    }

    bool operator!=(const Transition& other) const {
        return !(*this == other);
    }

private:
    const State& state_;
    const State& next_state_;
    const Action& action_;
    const Reward& reward_;
    Weight prob_weight_;
};

/**
 * Represents part of a transition, the next state & reward.
 *
 * Reward is stored by-value to allow fake/proxy rewards to be used.
 */
struct Response {
    /**
     * Create a Response from a Transition.
     *
     * This is made static as opposed to being a constructor so that we can maintain the
     * brace initialization.
     */
    static Response from_transition(const Transition& t) {
        Expects(t.prob_weight() >= 0);
        return Response{t.next_state(), t.reward(), t.prob_weight()};
    }

    const State& next_state;
    // note: I made Reward be stored by-value. This allows a Reward to be given that isn't a _real_
    // reward. This is an approach used group transitions that differ only in their reward into a
    // summary transition that has a reward equal to the expected value of the transitions being
    // grouped.
    Reward reward;
    Weight prob_weight;
};

/**
 * A container of Responses along with a weight total (sum of all response weights).
 *
 * Environment returns an object of this class to represent all possible transitions possible as a
 * result of an action in a state.
 */
class ResponseDistribution {
public:
    // It seems likely that this type should become a DistributionList so that a random response
    // can be obtained easily from a ResponseDistribution.
    using Responses = std::vector<Response>;

    const Responses& responses() const {return responses_;}
    Weight total_weight() const {return total_weight_;}

    void add_response(Response r) {
        Expects(r.prob_weight >= 0);
        total_weight_ += r.prob_weight;
        responses_.emplace_back(std::move(r));
    }
private:
    Responses responses_{};
    Weight total_weight_ = 0;
};

/**
 * Environment interface.
 *
 * No member variables and all methods are pure virtual.
 *
 * This interface was extracted from a previously concrete class (now called MappedEnvironment). The
 * interface was created so as to allow multiple implementations. The original implementation saved
 * all transition probabilities in a tree (DistributionTree). Later, a problem was encountered
 * (Jack's Car Rental) whereby the transition probabilities came from an analytical distribution
 * (Poisson distribution). In this case, it seems wasteful to pre-generate a transition tree as
 * opposed to creating transition lists on the fly.
 */
class Environment {
public:
    // Yuck. We still have std::unique_ptr and std::vector in the type. See DereferenceIterator.h
    // for more thoughts.
    // By using these types for our return values from methods such as states() we are imposing that
    // any implementation of Environment will need to store these objects on the heap. At the
    // moment this is an important requirement, as many classes pass around references/pointers to
    // these objects. We could switch to passing everything by value and the requirements on the
    // interface would be relaxed (although we would require more copying).
    using StateIterator =  util::DereferenceIterator<std::vector<std::unique_ptr<State>>::const_iterator>;
    using ActionIterator = util::DereferenceIterator<std::vector<std::unique_ptr<Action>>::const_iterator>;
    using RewardIterator = util::DereferenceIterator<std::vector<std::unique_ptr<Reward>>::const_iterator>;
    using States = util::RangeWrapper<StateIterator>;
    using Actions = util::RangeWrapper<ActionIterator>;

    //----------------------------------------------------------------------------------------------
    // Modify the environment
    //----------------------------------------------------------------------------------------------
    virtual void set_start_state(const State& state) = 0;
    // TODO: is this needed in the interface? You can change the start state, but I can't think of
    // a use-case to change the list of end states. It should be protected/private, I think.
    virtual void mark_as_end_state(const State& state) = 0;

    //----------------------------------------------------------------------------------------------
    // States
    //----------------------------------------------------------------------------------------------
    virtual ID state_count() const = 0;

    virtual const State& state(ID id) const = 0;
    virtual State& state(ID id) = 0;


    virtual StateIterator states_begin() const = 0;
    virtual StateIterator states_end() const = 0;

    virtual States states() const = 0;

    virtual std::vector<std::reference_wrapper<const State>> end_states() const = 0;

    virtual const State& start_state() const = 0;

    virtual bool is_end_state(const State& s) const = 0;

    //----------------------------------------------------------------------------------------------
    // Actions
    //----------------------------------------------------------------------------------------------
    virtual ID action_count() const = 0;
    virtual const Action& action(ID id) const = 0;
    virtual Action& action(ID id) = 0;

    virtual ActionIterator actions_begin() const = 0;
    virtual ActionIterator actions_end() const = 0;
    virtual Actions actions() const = 0;

    virtual bool is_action_allowed(const State& from_state, const Action& a) const = 0;

    //----------------------------------------------------------------------------------------------
    // Rewards
    //----------------------------------------------------------------------------------------------
    virtual ID reward_count() const = 0;
    virtual const Reward& reward(ID id) const = 0;
    virtual Reward& reward(ID id) = 0;

    virtual RewardIterator rewards_begin() const = 0;
    virtual RewardIterator rewards_end() const = 0;

    //----------------------------------------------------------------------------------------------
    // Transitions
    //----------------------------------------------------------------------------------------------
    // Random sample.
    /**
     *
     * \throws runtime_error if from_state is an end state. It is not possible to carry out any
     *         action in an end state.
     */
    virtual Response next_state(const State& from_state, const Action& action) const = 0;

    // Full MDP info.
    virtual ResponseDistribution transition_list(const State& from_state, const Action& action) const = 0;
};

} // namespace rl

// Following the examples here:
// https://en.cppreference.com/w/cpp/utility/hash
// Place the hash struct in the "std" namespace so that it is automatically chosen when creating
// maps.
namespace std {

template<> struct hash<rl::State>
{
    // Mark them as inline and we can keep them in the header.
    inline std::size_t operator()(const rl::State& s) const {
        return s.id();
    }
};

template<> struct hash<rl::Action>
{
    inline std::size_t operator()(const rl::Action& a) const {
        return a.id();
    }
};

template<> struct hash<rl::Reward>
{
    inline std::size_t operator()(const rl::Reward& r) const {
        return r.id();
    }
};

} // namespace std