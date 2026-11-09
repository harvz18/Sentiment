-- One-off, explicitly approved backend test workflow for the linked MULTIVENT project.
-- It preserves the standard selection, request, acceptance, payment, and completion
-- functions. Only the same-day event/payment guard is disabled transaction-locally.

begin;

do $$
declare
  event_id_value constant uuid := '5966a1d1-6303-427d-a398-adf62498ea94';
  client_id_value constant uuid := 'e1e431b4-a78f-44d0-a466-c700311b5442';
  provider_user_id_value constant uuid := 'a1100000-0000-4000-8000-000000000001';
  host_service_id constant uuid := 'ec68ebfc-193d-4d71-8106-98caea66b0dd';
  expected_services constant uuid[] := array[
    'e3d95f49-8c60-4e16-b1fd-76f01b4d7abf'::uuid, -- Aurora Grand Ballroom
    '2f2a4080-0b69-4c26-951f-5fe771013eb8'::uuid, -- Coastal Plated Celebration, no tasting
    'ec68ebfc-193d-4d71-8106-98caea66b0dd'::uuid  -- Arielle Wedding Host
  ];
  target_date constant date := date '2026-11-09';
  booking_id_value uuid;
  selection_result jsonb;
  payment_result jsonb;
  service_count integer;
  payment_count integer;
begin
  -- Work as the client for the existing event and use the normal host-selection RPC.
  perform set_config('request.jwt.claim.sub', client_id_value::text, true);
  perform set_config('request.jwt.claim.role', 'authenticated', true);

  if not exists (
    select 1 from public.events
    where id = event_id_value and client_id = client_id_value
      and status not in ('cancelled', 'completed')
  ) then
    raise exception 'Expected active client event is unavailable.';
  end if;

  if exists (
    select 1 from public.event_service_selections
    where event_id = event_id_value and service_id = host_service_id
      and status not in ('declined', 'cancelled')
  ) then
    raise exception 'Arielle Wedding Host is already selected; refusing to duplicate it.';
  end if;

  -- The client explicitly requested only the three provider services. Remove
  -- the pre-existing, unrelated coordinator invitation so it cannot become a
  -- fourth payment/acceptance requirement for this test event.
  perform set_config('app.coordinator_assignment_authorized', 'true', true);
  update public.events
  set wants_coordinator = false,
      coordinator_id = null,
      pending_coordinator_id = null,
      coordinator_assignment_status = null,
      coordinator_assignment_requested_at = null,
      coordinator_assignment_responded_at = null,
      coordinator_assignment_attempted_at = null,
      coordinator_assignment_note = 'Coordinator request withdrawn for the approved provider-only backend test workflow.',
      coordinator_fee_amount = 0,
      coordinator_pricing_snapshot = null,
      updated_at = now()
  where id = event_id_value;

  update public.coordinator_assignment_attempts
  set status = 'withdrawn',
      note = 'Withdrawn: the approved backend test workflow contains provider services only.',
      updated_at = now()
  where event_id = event_id_value and status in ('invited', 'pending');

  select public.save_my_event_service_selection(
    event_id_value, host_service_id, null, null, null, null,
    'One host/EMCEE requested through the approved backend test workflow.', false
  ) into selection_result;

  -- Existing venue/catering requests are retained. The standard request function
  -- adds only the new host booking and remains idempotent for the two existing rows.
  perform public.submit_my_event_booking_requests_v14(event_id_value);

  select count(*) into service_count
  from public.bookings
  where event_id = event_id_value
    and service_id = any(expected_services)
    and status not in ('cancelled', 'rejected', 'expired');
  if service_count <> 3 then
    raise exception 'Expected exactly three active provider bookings, found %.', service_count;
  end if;

  -- Accept each provider1 request through the standard provider decision RPC.
  perform set_config('request.jwt.claim.sub', provider_user_id_value::text, true);
  for booking_id_value in
    select id from public.bookings
    where event_id = event_id_value and service_id = any(expected_services)
      and status = 'requested'
    order by created_at, id
  loop
    perform public.respond_to_provider_booking(
      booking_id_value, true, 'Accepted by provider1 for the approved same-day backend test workflow.'
    );
  end loop;

  if exists (
    select 1 from public.bookings
    where event_id = event_id_value and service_id = any(expected_services)
      and status <> 'confirmed'
  ) then
    raise exception 'All three provider bookings must be confirmed before payment.';
  end if;

  -- Initiate the normal full test-payment records while the original future date
  -- remains in place, so all ordinary payment validation and ledger logic runs.
  perform set_config('request.jwt.claim.sub', client_id_value::text, true);
  select public.initiate_my_event_payment_v14(event_id_value, 'full', 'eWallet', true)
  into payment_result;

  select count(*) into payment_count
  from public.payments
  where event_id = event_id_value and booking_id in (
    select id from public.bookings
    where event_id = event_id_value and service_id = any(expected_services)
  ) and status in ('pending', 'processing');
  if payment_count <> 3 then
    raise exception 'Expected three pending full-payment records, found %.', payment_count;
  end if;

  -- Explicitly authorized one-off bypass: disable only the two authoritative
  -- same-day guards while changing this event and its newly-created payment rows.
  alter table public.events disable trigger zz_enforce_minimum_event_booking_lead_on_event;
  alter table public.payments disable trigger zz_enforce_minimum_event_booking_lead_on_initial_payment;

  update public.events
  set event_date = target_date,
      updated_at = now()
  where id = event_id_value;

  -- Keep the existing provider1 venue schedule aligned with the same-day event.
  update public.event_service_selections
  set venue_setup_start_at = timestamp '2026-11-09 13:00:00',
      venue_start_at = timestamp '2026-11-09 13:00:00',
      venue_end_at = timestamp '2026-11-09 23:00:00',
      venue_option_snapshot = jsonb_set(
        jsonb_set(jsonb_set(coalesce(venue_option_snapshot, '{}'::jsonb), '{eventDate}', to_jsonb('2026-11-09'::text), true),
          '{setupStartAt}', to_jsonb('2026-11-09T13:00:00'::text), true),
          '{startAt}', to_jsonb('2026-11-09T13:00:00'::text), true),
      venue_options_snapshot = jsonb_set(
        jsonb_set(jsonb_set(jsonb_set(coalesce(venue_options_snapshot, '[]'::jsonb), '{0,eventDate}', to_jsonb('2026-11-09'::text), true),
          '{0,setupStartAt}', to_jsonb('2026-11-09T13:00:00'::text), true),
          '{0,startAt}', to_jsonb('2026-11-09T13:00:00'::text), true),
          '{0,endAt}', to_jsonb('2026-11-09T23:00:00'::text), true),
      venue_booking_snapshot = jsonb_set(
        jsonb_set(jsonb_set(jsonb_set(jsonb_set(coalesce(venue_booking_snapshot, '{}'::jsonb), '{eventDate}', to_jsonb('2026-11-09'::text), true),
          '{selectedOptions,0,eventDate}', to_jsonb('2026-11-09'::text), true),
          '{selectedOptions,0,setupStartAt}', to_jsonb('2026-11-09T13:00:00'::text), true),
          '{selectedOptions,0,startAt}', to_jsonb('2026-11-09T13:00:00'::text), true),
          '{selectedOptions,0,endAt}', to_jsonb('2026-11-09T23:00:00'::text), true),
      updated_at = now()
  where event_id = event_id_value and service_id = expected_services[1];

  update public.bookings booking
  set requested_date = target_date,
      venue_setup_start_at = selection.venue_setup_start_at,
      venue_start_at = selection.venue_start_at,
      venue_end_at = selection.venue_end_at,
      venue_option_snapshot = selection.venue_option_snapshot,
      venue_options_snapshot = selection.venue_options_snapshot,
      venue_booking_snapshot = selection.venue_booking_snapshot,
      updated_at = now()
  from public.event_service_selections selection
  where booking.event_id = event_id_value
    and booking.event_id = selection.event_id
    and booking.service_id = selection.service_id
    and booking.service_id = any(expected_services);

  -- Confirm the capstone test payment. This is a test record, not a real charge.
  perform set_config('app.v14_payment_verification_authorized', 'true', true);
  update public.payments
  set status = 'verified', provider = 'eWallet',
      provider_reference = 'backend-test-payment-20261109',
      paid_at = coalesce(paid_at, now()), verified_at = now(),
      metadata = coalesce(metadata, '{}'::jsonb) || jsonb_build_object(
        'backendTestWorkflow', true,
        'backendTestConfirmedAt', now(),
        'backendTestSameDayBypassAuthorized', true
      ),
      updated_at = now()
  where event_id = event_id_value
    and booking_id in (
      select id from public.bookings
      where event_id = event_id_value and service_id = any(expected_services)
    ) and status in ('pending', 'processing');

  alter table public.payments enable trigger zz_enforce_minimum_event_booking_lead_on_initial_payment;
  alter table public.events enable trigger zz_enforce_minimum_event_booking_lead_on_event;

  -- Providers use the established completion RPC. With full payment confirmed
  -- and the event date now today, each call closes its service and the event.
  perform set_config('request.jwt.claim.sub', provider_user_id_value::text, true);
  for booking_id_value in
    select id from public.bookings
    where event_id = event_id_value and service_id = any(expected_services)
    order by created_at, id
  loop
    perform public.mark_provider_booking_completed(booking_id_value);
  end loop;

  if exists (
    select 1 from public.bookings
    where event_id = event_id_value and service_id = any(expected_services)
      and status <> 'completed'
  ) then
    raise exception 'One or more provider bookings did not reach completed status.';
  end if;

  if not exists (
    select 1 from public.events where id = event_id_value and status = 'completed'
  ) then
    raise exception 'The event did not reach completed status.';
  end if;
end;
$$;

commit;

select e.id as event_id, e.name, e.event_date, e.status as event_status,
  jsonb_agg(jsonb_build_object(
    'booking_id', b.id, 'service', s.name, 'booking_status', b.status,
    'requested_date', b.requested_date, 'food_tasting_preference', b.food_tasting_preference,
    'payment_status', p.status, 'provider_payment_status', b.completion_payment_status
  ) order by s.name) as bookings
from public.events e
join public.bookings b on b.event_id = e.id
join public.services s on s.id = b.service_id
left join public.payments p on p.booking_id = b.id and p.event_id = e.id
where e.id = '5966a1d1-6303-427d-a398-adf62498ea94'
group by e.id, e.name, e.event_date, e.status;
